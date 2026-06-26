"""
experiment_geometric_compound.py
==================================
Addresses two reviewer comments:

  [RC-A] Robustness against severe geometric transformations
         (rotation, scaling, flipping, and combined attacks)
  [RC-B] 3-way classification analysis under mixed / compound attacks

Runs all experiments on every image in grayscale_normalized/.

Outputs (saved in geometric_compound_results/)
-------
  Table_Geometric_Attacks.csv   → Supplementary Table S9
  Table_Compound_Attacks.csv    → Supplementary Table S10
  Table_Branch_Matrix.csv       → Branch-selection summary for text discussion

Run from:  C:/Users/tiwar/Downloads/journal implementation/
"""

import cv2
import os
import glob
import math
import csv
import hashlib
import numpy as np
from scipy import stats
from skimage.metrics import structural_similarity as ssim
import my_custom_method as watermark_system
import attack_image as attacker

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_DIR  = "grayscale_normalized"
OUT_DIR    = "geometric_compound_results"
BLOCK_SIZE = 4
KEY        = 9999
np.random.seed(42)

# ── Metric helpers ────────────────────────────────────────────────────────────
def psnr(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    mse = np.mean((a - b) ** 2)
    return 100.0 if mse < 1e-12 else 20 * math.log10(255.0 / math.sqrt(mse))

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

def tdr(original, attacked_wm, block_size=4):
    """
    Tamper Detection Rate = (detected tampered blocks) / (truly tampered blocks).
    'Truly tampered' = blocks where any pixel value differs between original
    (after watermark embedding re-embeds on same image) and attacked image.
    Since we compare watermarked vs attacked, any changed block is truly tampered.
    """
    h  = (original.shape[0] // block_size) * block_size
    w  = (original.shape[1] // block_size) * block_size
    orig_c  = original[:h, :w]
    atk_c   = attacked_wm[:h, :w]
    if orig_c.shape != atk_c.shape:
        atk_c = cv2.resize(atk_c, (w, h))

    total_changed, total_detected = 0, 0
    for ch in range(3):
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                ob = orig_c[ch if orig_c.ndim == 3 else ..., i:i+block_size, j:j+block_size] \
                     if orig_c.ndim == 2 else orig_c[i:i+block_size, j:j+block_size, ch]
                ab = atk_c[i:i+block_size, j:j+block_size, ch] \
                     if atk_c.ndim == 3 else atk_c[i:i+block_size, j:j+block_size]
                if np.any(ob != ab):
                    total_changed += 1
    return 100.0 * total_changed / max(1, total_changed), total_changed  # trivially 100% by def

def tamper_rate(img_bgr):
    """
    Fraction (%) of blocks flagged as tampered by the hash-check Pass-1.
    Replicates DLSBM Pass-1 detection without side effects.
    """
    img = img_bgr.copy()
    h, w = img.shape[:2]
    h = (h // BLOCK_SIZE) * BLOCK_SIZE
    w = (w // BLOCK_SIZE) * BLOCK_SIZE
    img = img[:h, :w]
    total  = (h // BLOCK_SIZE) * (w // BLOCK_SIZE)
    flagged = 0
    for ch in range(3):
        channel = img[:, :, ch]
        idx = 0
        for i in range(0, h, BLOCK_SIZE):
            for j in range(0, w, BLOCK_SIZE):
                blk  = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
                flat = blk.flatten()
                bits = "".join(str((v >> b) & 1) for v in flat for b in [0, 1])
                clean = (blk & 0xFC).flatten()
                h_data = hashlib.md5(
                    clean.tobytes() + int(idx).to_bytes(4, 'big')
                ).hexdigest()
                expected = f"{int(h_data[:3], 16):012b}"
                if expected != bits[:12]:
                    flagged += 1
                idx += 1
    return 100.0 * flagged / (total * 3)

def classify_branch(img_bgr):
    """
    Return (tau_pct, eta_pct, branch_label) without running the full recover().
    Mirrors the DLSBM 3-way classification logic.
    """
    img = img_bgr.copy()
    h, w = img.shape[:2]
    h = (h // BLOCK_SIZE) * BLOCK_SIZE
    w = (w // BLOCK_SIZE) * BLOCK_SIZE
    img = img[:h, :w]

    tau = tamper_rate(img)

    # eta = fraction of extreme pixels (0 or 255) — salt-and-pepper indicator
    extreme = np.sum((img == 0) | (img == 255))
    eta = 100.0 * extreme / (h * w * 3)

    is_noise = (eta > 0.5)           # η > 0.5 %
    is_jpeg  = (tau > 85.0) and not is_noise

    if is_jpeg:
        branch = "A"
    elif is_noise:
        branch = "B"
    else:
        branch = "C"

    return tau, eta, branch

def ci95(values):
    n  = len(values)
    m  = float(np.mean(values))
    s  = float(np.std(values, ddof=1))
    hw = float(stats.t.ppf(0.975, df=n - 1) * s / math.sqrt(n))
    return m, s, hw

# ── Geometric attacks ─────────────────────────────────────────────────────────
def geo_rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def geo_scale(img, factor):
    """Resample to factor× then back to original size."""
    h, w = img.shape[:2]
    nh, nw = max(4, int(h * factor)), max(4, int(w * factor))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

def geo_flip(img, code):
    return cv2.flip(img, code)

def geo_combined(img, angle, factor):
    return geo_scale(geo_rotate(img, angle), factor)

GEOMETRIC_ATTACKS = [
    # (display label,                    function,       args)
    ("Rotation $5°$",                    geo_rotate,     (5,)),
    ("Rotation $15°$",                   geo_rotate,     (15,)),
    ("Rotation $30°$",                   geo_rotate,     (30,)),
    ("Rotation $45°$",                   geo_rotate,     (45,)),
    ("Rotation $90°$",                   geo_rotate,     (90,)),
    ("Scaling $\\times 0.50$",           geo_scale,      (0.50,)),
    ("Scaling $\\times 0.75$",           geo_scale,      (0.75,)),
    ("Scaling $\\times 1.25$",           geo_scale,      (1.25,)),
    ("Scaling $\\times 1.50$",           geo_scale,      (1.50,)),
    ("Horizontal Flip",                  geo_flip,       (1,)),
    ("Vertical Flip",                    geo_flip,       (0,)),
    ("Rot $15°$ + Scale $\\times 0.90$", geo_combined,   (15, 0.90)),
    ("Rot $30°$ + Scale $\\times 0.80$", geo_combined,   (30, 0.80)),
]

# ── Compound attacks ──────────────────────────────────────────────────────────
def cmp_jpeg_noise(img, quality=90, amount=0.03):
    t, _ = attacker.attack_jpeg_compression(img.copy(), quality)
    t, _ = attacker.attack_salt_and_pepper(t, amount)
    return t

def cmp_jpeg_noise_q70(img):
    return cmp_jpeg_noise(img, quality=70, amount=0.05)

def cmp_crop_jpeg(img, pct=30, quality=70):
    t, _ = attacker.attack_cropping(img.copy(), percent=pct)
    t, _ = attacker.attack_jpeg_compression(t, quality)
    return t

def cmp_crop_noise(img, pct=30, amount=0.05):
    t, _ = attacker.attack_cropping(img.copy(), percent=pct)
    t, _ = attacker.attack_salt_and_pepper(t, amount)
    return t

def cmp_struct_jpeg(img, quality=90):
    t, _ = attacker.attack_content_removal(img.copy())
    t, _ = attacker.attack_jpeg_compression(t, quality)
    return t

def cmp_struct_noise(img, amount=0.05):
    t, _ = attacker.attack_content_removal(img.copy())
    t, _ = attacker.attack_salt_and_pepper(t, amount)
    return t

def cmp_copymove_jpeg(img, quality=70):
    t, _ = attacker.attack_copy_move(img.copy())
    t, _ = attacker.attack_jpeg_compression(t, quality)
    return t

def cmp_copymove_noise(img, amount=0.05):
    t, _ = attacker.attack_copy_move(img.copy())
    t, _ = attacker.attack_salt_and_pepper(t, amount)
    return t

def cmp_noise_crop(img, amount=0.05, pct=20):
    """Order matters: noise first then structural crop."""
    t, _ = attacker.attack_salt_and_pepper(img.copy(), amount)
    t, _ = attacker.attack_cropping(t, percent=pct)
    return t

COMPOUND_ATTACKS = [
    # (display label,                        function,          args)
    ("JPEG($Q$=90) + S\\&P(3\\%)",           cmp_jpeg_noise,    (90, 0.03)),
    ("JPEG($Q$=70) + S\\&P(5\\%)",           cmp_jpeg_noise_q70,()),
    ("Crop(30\\%) + JPEG($Q$=70)",           cmp_crop_jpeg,     (30, 70)),
    ("Crop(30\\%) + S\\&P(5\\%)",            cmp_crop_noise,    (30, 0.05)),
    ("Content Removal + JPEG($Q$=90)",       cmp_struct_jpeg,   (90,)),
    ("Content Removal + S\\&P(5\\%)",        cmp_struct_noise,  (0.05,)),
    ("Copy-Move + JPEG($Q$=70)",             cmp_copymove_jpeg, (70,)),
    ("Copy-Move + S\\&P(5\\%)",              cmp_copymove_noise,(0.05,)),
    ("S\\&P(5\\%) $\\rightarrow$ Crop(20\\%)",cmp_noise_crop,   (0.05, 20)),
]

# ── I/O helpers ───────────────────────────────────────────────────────────────
def setup_dirs():
    for sub in ["watermarked",
                "geometric/attacked", "geometric/recovered",
                "compound/attacked",  "compound/recovered"]:
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

def get_files():
    files = []
    for ext in ("*.tiff", "*.png", "*.jpg", "*.jpeg"):
        files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    return sorted(files)

# ── Run one experiment ────────────────────────────────────────────────────────
def run_experiment(label, attacked_img, orig_img, wm_img, rec_dir, base, atk_idx):
    atk_path = os.path.join(rec_dir.replace("recovered", "attacked"),
                            f"{base}_{atk_idx:02d}.png")
    rec_path = os.path.join(rec_dir, f"{base}_{atk_idx:02d}.png")

    cv2.imwrite(atk_path, attacked_img)

    # Classify branch BEFORE recovery
    tau, eta, branch = classify_branch(attacked_img)

    # Run full recovery
    watermark_system.recover(atk_path, rec_path)
    rec = cv2.imread(rec_path)

    rp = psnr(orig_img, rec)
    rs = compute_ssim(orig_img, rec)
    rm = compute_msssim(orig_img, rec)

    return tau, eta, branch, rp, rs, rm

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    setup_dirs()
    files = get_files()
    if not files:
        print(f"ERROR: no images found in '{INPUT_DIR}/'"); return

    print("=" * 72)
    print("  DLSBM – Geometric & Compound Attack Experiments")
    print(f"  Images: {len(files)}  |  Geometric: {len(GEOMETRIC_ATTACKS)}  "
          f"|  Compound: {len(COMPOUND_ATTACKS)}")
    print("=" * 72)

    # ── Step 1: Embed watermarks ──────────────────────────────────────────────
    print("\n[Phase 1] Embedding watermarks ...")
    wm_paths = {}
    orig_imgs = {}
    for fpath in files:
        base    = os.path.splitext(os.path.basename(fpath))[0]
        wm_path = os.path.join(OUT_DIR, "watermarked", f"{base}.png")
        orig_imgs[base] = cv2.imread(fpath)
        if not os.path.exists(wm_path):
            watermark_system.embed(fpath, wm_path)
        wm_paths[base] = wm_path
        print(f"  {base}")

    # ── Step 2: Geometric attacks ─────────────────────────────────────────────
    print("\n[Phase 2] Geometric attack experiments ...")
    # geo_results[attack_label] = list of (tau, eta, branch, rp, rs, rm) per image
    geo_results = {label: [] for label, _, _ in GEOMETRIC_ATTACKS}

    for fpath in files:
        base    = os.path.splitext(os.path.basename(fpath))[0]
        orig    = orig_imgs[base]
        wm      = cv2.imread(wm_paths[base])
        rec_dir = os.path.join(OUT_DIR, "geometric", "recovered")

        print(f"\n  [{base}]")
        for atk_idx, (label, fn, args) in enumerate(GEOMETRIC_ATTACKS):
            attacked = fn(wm.copy(), *args)
            tau, eta, branch, rp, rs, rm = run_experiment(
                label, attacked, orig, wm, rec_dir, base, atk_idx)
            geo_results[label].append((tau, eta, branch, rp, rs, rm))
            print(f"    {label:<38} tau={tau:5.1f}% eta={eta:5.2f}% "
                  f"Br={branch}  R-PSNR={rp:.2f} SSIM={rs:.4f}")

    # ── Step 3: Compound attacks ──────────────────────────────────────────────
    print("\n[Phase 3] Compound attack experiments ...")
    cmp_results = {label: [] for label, _, _ in COMPOUND_ATTACKS}

    for fpath in files:
        base    = os.path.splitext(os.path.basename(fpath))[0]
        orig    = orig_imgs[base]
        wm      = cv2.imread(wm_paths[base])
        rec_dir = os.path.join(OUT_DIR, "compound", "recovered")

        print(f"\n  [{base}]")
        for atk_idx, (label, fn, args) in enumerate(COMPOUND_ATTACKS):
            attacked = fn(wm.copy(), *args)
            tau, eta, branch, rp, rs, rm = run_experiment(
                label, attacked, orig, wm, rec_dir, base, atk_idx)
            cmp_results[label].append((tau, eta, branch, rp, rs, rm))
            print(f"    {label:<42} tau={tau:5.1f}% eta={eta:5.2f}% "
                  f"Br={branch}  R-PSNR={rp:.2f} SSIM={rs:.4f}")

    # ════════════════════════════════════════════════════════════════════════
    # CSV 1 – Geometric attacks  (→ Table S9)
    # ════════════════════════════════════════════════════════════════════════
    geo_csv = os.path.join(OUT_DIR, "Table_Geometric_Attacks.csv")
    with open(geo_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Attack",
                    "tau mean (%)", "tau std",
                    "eta mean (%)", "eta std",
                    "Branch (mode)",
                    "R-PSNR mean", "R-PSNR std", "R-PSNR CI+-",
                    "SSIM mean",   "SSIM std",   "SSIM CI+-",
                    "MS-SSIM mean"])
        for label, _, _ in GEOMETRIC_ATTACKS:
            rows = geo_results[label]
            taus  = [r[0] for r in rows]
            etas  = [r[1] for r in rows]
            brs   = [r[2] for r in rows]
            rps   = [r[3] for r in rows]
            rss   = [r[4] for r in rows]
            rms   = [r[5] for r in rows]

            # mode branch
            branch_mode = max(set(brs), key=brs.count)
            branches_str = "/".join(sorted(set(brs)))

            tm, ts, _ = ci95(taus)
            em, es, _ = ci95(etas)
            pm, ps, pc = ci95(rps)
            sm, ss, sc = ci95(rss)
            mm, ms, _  = ci95(rms)

            w.writerow([label,
                        f"{tm:.1f}", f"{ts:.1f}",
                        f"{em:.2f}", f"{es:.2f}",
                        branch_mode,
                        f"{pm:.2f}", f"{ps:.2f}", f"{pc:.2f}",
                        f"{sm:.4f}", f"{ss:.4f}", f"{sc:.4f}",
                        f"{mm:.4f}"])

    # ════════════════════════════════════════════════════════════════════════
    # CSV 2 – Compound attacks  (→ Table S10)
    # ════════════════════════════════════════════════════════════════════════
    cmp_csv = os.path.join(OUT_DIR, "Table_Compound_Attacks.csv")
    with open(cmp_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Compound Attack",
                    "tau mean (%)", "tau std",
                    "eta mean (%)", "eta std",
                    "Branch (mode)",
                    "R-PSNR mean", "R-PSNR std", "R-PSNR CI+-",
                    "SSIM mean",   "SSIM std",   "SSIM CI+-",
                    "MS-SSIM mean"])
        for label, _, _ in COMPOUND_ATTACKS:
            rows = cmp_results[label]
            taus  = [r[0] for r in rows]
            etas  = [r[1] for r in rows]
            brs   = [r[2] for r in rows]
            rps   = [r[3] for r in rows]
            rss   = [r[4] for r in rows]
            rms   = [r[5] for r in rows]

            branch_mode = max(set(brs), key=brs.count)

            tm, ts, _ = ci95(taus)
            em, es, _ = ci95(etas)
            pm, ps, pc = ci95(rps)
            sm, ss, sc = ci95(rss)
            mm, ms, _  = ci95(rms)

            w.writerow([label,
                        f"{tm:.1f}", f"{ts:.1f}",
                        f"{em:.2f}", f"{es:.2f}",
                        branch_mode,
                        f"{pm:.2f}", f"{ps:.2f}", f"{pc:.2f}",
                        f"{sm:.4f}", f"{ss:.4f}", f"{sc:.4f}",
                        f"{mm:.4f}"])

    # ════════════════════════════════════════════════════════════════════════
    # CSV 3 – Branch selection matrix  (all 9 images × all attacks)
    # ════════════════════════════════════════════════════════════════════════
    brm_csv = os.path.join(OUT_DIR, "Table_Branch_Matrix.csv")
    image_names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    with open(brm_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Attack"] + image_names + ["Consensus"])
        for label, _, _ in COMPOUND_ATTACKS:
            brs = [r[2] for r in cmp_results[label]]
            consensus = max(set(brs), key=brs.count)
            w.writerow([label] + brs + [consensus])

    # ════════════════════════════════════════════════════════════════════════
    # Console summary
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("GEOMETRIC ATTACKS SUMMARY")
    print("=" * 72)
    print(f"{'Attack':<42} {'tau%':>7} {'Br':>4} {'R-PSNR':>8} {'SSIM':>7} {'MS-SSIM':>9}")
    print("-" * 72)
    for label, _, _ in GEOMETRIC_ATTACKS:
        rows = geo_results[label]
        taus = [r[0] for r in rows]
        brs  = [r[2] for r in rows]
        rps  = [r[3] for r in rows]
        rss  = [r[4] for r in rows]
        rms  = [r[5] for r in rows]
        branch_mode = max(set(brs), key=brs.count)
        pm, ps, _ = ci95(rps)
        sm, ss, _ = ci95(rss)
        mm, _, _  = ci95(rms)
        tm = float(np.mean(taus))
        print(f"  {label:<40} {tm:>5.1f}% {branch_mode:>5}  {pm:>7.2f}  {sm:>7.4f}  {mm:>9.4f}")

    print("\n" + "=" * 72)
    print("COMPOUND ATTACKS SUMMARY")
    print("=" * 72)
    print(f"{'Compound Attack':<46} {'tau%':>7} {'eta%':>5} {'Br':>4} {'R-PSNR':>8} {'SSIM':>7}")
    print("-" * 72)
    for label, _, _ in COMPOUND_ATTACKS:
        rows = cmp_results[label]
        taus  = [r[0] for r in rows]
        etas  = [r[1] for r in rows]
        brs   = [r[2] for r in rows]
        rps   = [r[3] for r in rows]
        rss   = [r[4] for r in rows]
        branch_mode = max(set(brs), key=brs.count)
        pm, _, _ = ci95(rps)
        sm, _, _ = ci95(rss)
        tm = float(np.mean(taus))
        em = float(np.mean(etas))
        print(f"  {label:<44} {tm:>5.1f}% {em:>4.1f}% {branch_mode:>5}  {pm:>7.2f}  {sm:>7.4f}")

    print(f"\nOutput files:")
    print(f"  {geo_csv}")
    print(f"  {cmp_csv}")
    print(f"  {brm_csv}")


if __name__ == "__main__":
    main()
