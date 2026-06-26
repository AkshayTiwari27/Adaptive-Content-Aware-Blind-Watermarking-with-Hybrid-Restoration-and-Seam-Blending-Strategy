"""
compute_timing_and_stats.py
============================
Addresses three reviewer comments:

  RC-3  : Report execution time of embedding, detection, and recovery stages
  RC-5  : Add standard deviation and 95% confidence intervals to all tables
  RC-perceptual : Add MS-SSIM as an additional perceptual quality metric

Outputs
-------
  Table_Timing.csv        -- per-image embed / detect / recovery times + mean±std, CI
  Table_StdCI_WPQ.csv     -- Table 1 (W-PSNR, SSIM) with std and 95% CI
  Table_StdCI_Recovery.csv-- Table 3 (R-PSNR, SSIM, MS-SSIM) with std and 95% CI
  Table_MSSSIM.csv        -- attack-level MS-SSIM summary

Run from:  C:/Users/tiwar/Downloads/journal implementation/
"""

import cv2
import os
import glob
import time
import math
import csv
import hashlib
import numpy as np
from scipy import stats
from skimage.metrics import structural_similarity as ssim
import my_custom_method as watermark_system
import attack_image as attacker

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_DIR  = "grayscale_normalized"
TIMING_DIR = "timing_results"
BLOCK_SIZE = 4
KEY        = 9999

ATTACK_TYPES = [
    "content_removal", "copy_move", "splicing",
    "jpeg_compression", "noise", "cropping"
]

ATTACK_LABELS = {
    "content_removal" : "Content Removal",
    "copy_move"       : "Copy-Move",
    "splicing"        : "Splicing",
    "jpeg_compression": "JPEG ($Q=90$)",
    "noise"           : "Salt \\& Pepper (0.05)",
    "cropping"        : "Cropping (40\\%)",
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def psnr(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    mse = np.mean((a - b) ** 2)
    return 100.0 if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse))

def compute_ssim(a, b):
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    ax = 2 if a.ndim == 3 else None
    return float(ssim(a, b, data_range=255, channel_axis=ax))

def compute_msssim(a, b, weights=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
    """Multi-Scale SSIM (Wang et al., 2003)."""
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    scores, used_w = [], []
    ax = 2 if a.ndim == 3 else None
    for w in weights:
        if a.shape[0] < 16 or a.shape[1] < 16:
            break
        scores.append(ssim(a, b, data_range=255, channel_axis=ax))
        used_w.append(w)
        a = cv2.resize(a, (a.shape[1] // 2, a.shape[0] // 2))
        b = cv2.resize(b, (b.shape[1] // 2, b.shape[0] // 2))
    if not scores:
        return 0.0
    total = sum(used_w)
    return float(np.prod([s ** (w / total) for s, w in zip(scores, used_w)]))

def ci95(values):
    """Return (mean, std, half-width of 95% CI) using t-distribution."""
    n = len(values)
    m  = float(np.mean(values))
    s  = float(np.std(values, ddof=1))
    hw = float(stats.t.ppf(0.975, df=n - 1) * s / math.sqrt(n))
    return m, s, hw

def fmt(m, s, hw, nd=2):
    return f"{m:.{nd}f} ± {s:.{nd}f} (CI: ±{hw:.{nd}f})"

# ── Detection Pass-1 (timing only, no side-effects) ─────────────────────────
def _get_loc_hash(flat_block, idx):
    data = flat_block.tobytes()
    ib   = int(idx).to_bytes(4, byteorder='big')
    h    = hashlib.md5(data + ib).hexdigest()
    return f"{int(h[:3], 16):012b}"

def time_detection_only(img_bgr):
    """Replay Pass 1 of recover(); return elapsed seconds."""
    img = img_bgr.copy()
    h, w = img.shape[:2]
    h = (h // BLOCK_SIZE) * BLOCK_SIZE
    w = (w // BLOCK_SIZE) * BLOCK_SIZE
    img = img[:h, :w]

    t0 = time.perf_counter()
    watermark_system.get_smart_mapping(h, w, BLOCK_SIZE, KEY)   # mapping build
    for ch in range(3):
        channel = img[:, :, ch]
        idx = 0
        for i in range(0, h, BLOCK_SIZE):
            for j in range(0, w, BLOCK_SIZE):
                blk  = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
                flat = blk.flatten()
                bits = ""
                for k in range(16):
                    v = flat[k]
                    bits += str(v & 1) + str((v >> 1) & 1)
                clean = (blk & 0xFC)
                _get_loc_hash(clean.flatten(), idx)
                idx += 1
    return time.perf_counter() - t0

# ── Setup ────────────────────────────────────────────────────────────────────
def setup_dirs():
    os.makedirs(os.path.join(TIMING_DIR, "watermarked"), exist_ok=True)
    for atk in ATTACK_TYPES:
        os.makedirs(os.path.join(TIMING_DIR, atk, "attacked"),  exist_ok=True)
        os.makedirs(os.path.join(TIMING_DIR, atk, "recovered"), exist_ok=True)

def perform_attack(name, img):
    if name == "content_removal" : return attacker.attack_content_removal(img)
    if name == "copy_move"       : return attacker.attack_copy_move(img)
    if name == "splicing"        : return attacker.attack_political_splicing(img)
    if name == "cropping"        : return attacker.attack_cropping(img, percent=40)
    if name == "jpeg_compression": return attacker.attack_jpeg_compression(img, quality=90)
    if name == "noise"           : return attacker.attack_salt_and_pepper(img, amount=0.05)
    return None, None

def get_files():
    files = []
    for ext in ["*.tiff", "*.png", "*.jpg", "*.jpeg"]:
        files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    return sorted(files)

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    setup_dirs()
    files = get_files()
    if not files:
        print(f"ERROR: no images found in {INPUT_DIR}/"); return

    print("=" * 68)
    print("  DLSBM  –  Timing + Statistical Analysis + MS-SSIM")
    print("=" * 68)

    # ── Containers ──
    embed_t   = []                              # per-image embedding times
    detect_t  = {a: [] for a in ATTACK_TYPES}  # per-image detection times
    recover_t = {a: [] for a in ATTACK_TYPES}  # per-image recovery times

    wpsnr_vals = []
    wssim_vals = []

    rpsnr_vals = {a: [] for a in ATTACK_TYPES}
    rssim_vals = {a: [] for a in ATTACK_TYPES}
    msssim_vals= {a: [] for a in ATTACK_TYPES}

    timing_rows = []   # detailed rows for CSV

    for fpath in files:
        base    = os.path.splitext(os.path.basename(fpath))[0]
        wm_path = os.path.join(TIMING_DIR, "watermarked", f"{base}.png")
        orig    = cv2.imread(fpath)
        print(f"\n[{base}]")

        # ── Embedding ──
        t0 = time.perf_counter()
        watermark_system.embed(fpath, wm_path)
        te = time.perf_counter() - t0
        embed_t.append(te)

        wm = cv2.imread(wm_path)
        wpsnr_vals.append(psnr(orig, wm))
        wssim_vals.append(compute_ssim(orig, wm))
        print(f"  embed: {te:.3f}s")

        # ── Per-attack: detect + recover ──
        for atk in ATTACK_TYPES:
            atk_path = os.path.join(TIMING_DIR, atk, "attacked",  f"{base}.png")
            rec_path = os.path.join(TIMING_DIR, atk, "recovered", f"{base}.png")

            attacked, _ = perform_attack(atk, wm.copy())
            if attacked is None:
                continue
            cv2.imwrite(atk_path, attacked)

            # Detection time (Pass 1 only)
            td = time_detection_only(attacked)

            # Full recovery time
            t0 = time.perf_counter()
            watermark_system.recover(atk_path, rec_path)
            t_full = time.perf_counter() - t0

            # Recovery-only ≈ full − detection
            tr = max(0.0, t_full - td)

            detect_t[atk].append(td)
            recover_t[atk].append(tr)

            rec = cv2.imread(rec_path)
            rp  = psnr(orig, rec)
            rs  = compute_ssim(orig, rec)
            rm  = compute_msssim(orig, rec)
            rpsnr_vals[atk].append(rp)
            rssim_vals[atk].append(rs)
            msssim_vals[atk].append(rm)

            timing_rows.append([base, ATTACK_LABELS[atk],
                                 f"{te:.4f}", f"{td:.4f}", f"{tr:.4f}"])
            print(f"  {atk:20s}  det={td:.3f}s  rec={tr:.3f}s  "
                  f"R-PSNR={rp:.2f}dB  SSIM={rs:.4f}  MS-SSIM={rm:.4f}")

    # ════════════════════════════════════════════════════════════════════════
    # CSV 1 – Timing
    # ════════════════════════════════════════════════════════════════════════
    with open("Table_Timing.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Image", "Attack", "Embed (s)", "Detection (s)", "Recovery (s)"])
        w.writerows(timing_rows)
        w.writerow([])

        # Summary
        em, es, ec = ci95(embed_t)
        w.writerow(["AVERAGE ± STD (95% CI)", "Embedding",
                    fmt(em, es, ec, 3), "", ""])

        for atk in ATTACK_TYPES:
            if not detect_t[atk]: continue
            dm, ds, dc = ci95(detect_t[atk])
            rm, rs2, rc = ci95(recover_t[atk])
            w.writerow(["", ATTACK_LABELS[atk], "",
                        fmt(dm, ds, dc, 3), fmt(rm, rs2, rc, 3)])

    # ════════════════════════════════════════════════════════════════════════
    # CSV 2 – W-PSNR / SSIM with std + CI  (Table 1 extension)
    # ════════════════════════════════════════════════════════════════════════
    pm, ps, pc = ci95(wpsnr_vals)
    sm, ss2, sc = ci95(wssim_vals)

    with open("Table_StdCI_WPQ.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Mean", "Std", "95% CI Lower", "95% CI Upper"])
        w.writerow(["W-PSNR (dB)", f"{pm:.2f}", f"{ps:.2f}",
                    f"{pm-pc:.2f}", f"{pm+pc:.2f}"])
        w.writerow(["SSIM", f"{sm:.4f}", f"{ss2:.4f}",
                    f"{sm-sc:.4f}", f"{sm+sc:.4f}"])

    # ════════════════════════════════════════════════════════════════════════
    # CSV 3 – R-PSNR / SSIM / MS-SSIM with std + CI  (Table 3 extension)
    # ════════════════════════════════════════════════════════════════════════
    with open("Table_StdCI_Recovery.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Attack",
                    "R-PSNR mean", "R-PSNR std", "R-PSNR CI±",
                    "SSIM mean",   "SSIM std",   "SSIM CI±",
                    "MS-SSIM mean","MS-SSIM std","MS-SSIM CI±"])
        for atk in ATTACK_TYPES:
            if not rpsnr_vals[atk]: continue
            pm2, ps2, pc2 = ci95(rpsnr_vals[atk])
            sm2, ss3, sc2 = ci95(rssim_vals[atk])
            mm,  ms2, mc  = ci95(msssim_vals[atk])
            w.writerow([ATTACK_LABELS[atk],
                        f"{pm2:.2f}", f"{ps2:.2f}", f"{pc2:.2f}",
                        f"{sm2:.4f}", f"{ss3:.4f}", f"{sc2:.4f}",
                        f"{mm:.4f}",  f"{ms2:.4f}", f"{mc:.4f}"])

    # ════════════════════════════════════════════════════════════════════════
    # Console summary
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    em, es, ec = ci95(embed_t)
    print(f"\nEmbedding   : {em:.3f} ± {es:.3f} s   (95% CI: ±{ec:.3f})")
    print(f"\nW-PSNR      : {pm:.2f} ± {ps:.2f} dB  (95% CI: [{pm-pc:.2f}, {pm+pc:.2f}])")
    print(f"W-SSIM      : {sm:.4f} ± {ss2:.4f}    (95% CI: [{sm-sc:.4f}, {sm+sc:.4f}])")
    print()
    print(f"{'Attack':<24} {'Det(s)':>10} {'Rec(s)':>10} "
          f"{'R-PSNR':>8} {'SSIM':>7} {'MS-SSIM':>9}")
    print("-" * 68)
    for atk in ATTACK_TYPES:
        if not detect_t[atk]: continue
        dm, ds, dc   = ci95(detect_t[atk])
        rm2, rs2, rc = ci95(recover_t[atk])
        pm3, _,  _   = ci95(rpsnr_vals[atk])
        sm3, _,  _   = ci95(rssim_vals[atk])
        mm3, _,  _   = ci95(msssim_vals[atk])
        print(f"{ATTACK_LABELS[atk]:<24} {dm:>6.3f}±{ds:.3f} "
              f"{rm2:>6.3f}±{rs2:.3f} "
              f"{pm3:>8.2f} {sm3:>7.4f} {mm3:>9.4f}")

    print("\nOutput files written:")
    print("  Table_Timing.csv")
    print("  Table_StdCI_WPQ.csv")
    print("  Table_StdCI_Recovery.csv")


if __name__ == "__main__":
    main()
