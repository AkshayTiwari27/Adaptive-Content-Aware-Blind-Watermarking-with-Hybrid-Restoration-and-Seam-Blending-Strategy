"""
experiment_realworld.py
=======================
Validates DLSBM on real-world photographs outside the USC-SIPI benchmark.

Images : Kodak Lossless True Color Image Suite (24 uncompressed natural
         photographs widely used in image-quality research).
         5 images are selected to span diverse content categories:
         portrait, natural landscape, urban scene, indoor objects, close-up.

Tampering scenarios (4 practical real-world forgery types):
  1. credential_forgery  -- zeroes a 40x40 patch (text/stamp erasure)
  2. copy_move           -- clones a 60x60 region to another position
  3. splicing            -- inserts 80x80 patch from a different image
  4. social_media_jpeg   -- JPEG re-compression at Q=80 (sharing pipeline)

Output:
  realworld_results/Table_RealWorld.csv   (Supplementary Table S14)
  realworld_results/<name>_<attack>_*.png

Run from:  C:/Users/tiwar/Downloads/journal implementation/
"""

import os, sys, csv, math, urllib.request, hashlib
import cv2
import numpy as np
from scipy import stats
from skimage.metrics import structural_similarity as ssim_fn
import my_custom_method as wm

# ── Config ────────────────────────────────────────────────────────────────────
BLOCK_SIZE = 4
KEY        = 9999
OUT_DIR    = "realworld_results"
TARGET_H   = 512
TARGET_W   = 512

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "raw"), exist_ok=True)

# 5 Kodak images — diverse real-world content categories
KODAK_IMAGES = {
    "portrait":   "http://r0k.us/graphics/kodak/kodak/kodim04.png",
    "landscape":  "http://r0k.us/graphics/kodak/kodak/kodim13.png",
    "urban":      "http://r0k.us/graphics/kodak/kodak/kodim15.png",
    "indoor":     "http://r0k.us/graphics/kodak/kodak/kodim07.png",
    "closeup":    "http://r0k.us/graphics/kodak/kodak/kodim23.png",
}

# ── Metric helpers (match other experiment scripts exactly) ───────────────────
def psnr(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return 100.0 if mse < 1e-12 else 20 * math.log10(255.0 / math.sqrt(mse))

def compute_ssim(a, b):
    ax = 2 if a.ndim == 3 else None
    return float(ssim_fn(a, b, data_range=255, channel_axis=ax))

def compute_msssim(a, b, weights=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
    scores, used_w = [], []
    ax = 2 if a.ndim == 3 else None
    for w in weights:
        if a.shape[0] < 16 or a.shape[1] < 16:
            break
        scores.append(ssim_fn(a, b, data_range=255, channel_axis=ax))
        used_w.append(w)
        a = cv2.resize(a, (a.shape[1] // 2, a.shape[0] // 2))
        b = cv2.resize(b, (b.shape[1] // 2, b.shape[0] // 2))
    if not scores:
        return 0.0
    total = sum(used_w)
    return float(np.prod([s ** (wt / total) for s, wt in zip(scores, used_w)]))

def ci95(values):
    n = len(values)
    m = float(np.mean(values))
    if n < 2:
        return m, 0.0, 0.0
    se = float(np.std(values, ddof=1)) / math.sqrt(n)
    t  = stats.t.ppf(0.975, df=n - 1)
    return m, float(np.std(values, ddof=1)), t * se

# ── classify_branch (mirrors DLSBM Pass-1, no side effects) ──────────────────
def _tamper_rate(img_bgr):
    img = img_bgr.copy()
    h, w = img.shape[:2]
    h = (h // BLOCK_SIZE) * BLOCK_SIZE
    w = (w // BLOCK_SIZE) * BLOCK_SIZE
    img = img[:h, :w]
    total   = (h // BLOCK_SIZE) * (w // BLOCK_SIZE)
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
    img = img_bgr.copy()
    h, w = img.shape[:2]
    h = (h // BLOCK_SIZE) * BLOCK_SIZE
    w = (w // BLOCK_SIZE) * BLOCK_SIZE
    img = img[:h, :w]
    tau     = _tamper_rate(img)
    extreme = np.sum((img == 0) | (img == 255))
    eta     = 100.0 * extreme / (h * w * 3)
    is_noise = (eta  > 0.5)
    is_jpeg  = (tau  > 85.0) and not is_noise
    branch   = "A" if is_jpeg else ("B" if is_noise else "C")
    return tau, eta, branch

# ── Image download & preprocessing ───────────────────────────────────────────
def download(url, dest):
    if os.path.exists(dest):
        return True
    try:
        print(f"  Downloading {os.path.basename(dest)} ...")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  FAILED ({e})")
        return False

def load_crop_resize(path, h=TARGET_H, w=TARGET_W):
    img = cv2.imread(path)
    if img is None:
        return None
    ih, iw = img.shape[:2]
    s   = min(ih, iw)
    img = img[(ih - s)//2:(ih - s)//2 + s, (iw - s)//2:(iw - s)//2 + s]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)

# ── Tampering functions (real-world forgery scenarios) ────────────────────────
def atk_credential_forgery(img):
    """Erase a 40x40 region (text/stamp removal on a scanned document)."""
    out = img.copy()
    h, w = out.shape[:2]
    y, x = h // 3, w // 3
    out[y:y+40, x:x+40] = 0
    return out

def atk_copy_move(img):
    """Clone a 60x60 region to a new position (background duplication)."""
    out = img.copy()
    h, w = out.shape[:2]
    out[h//2:h//2+60, w//2:w//2+60] = out[h//5:h//5+60, w//5:w//5+60]
    return out

def atk_splicing(img, donor):
    """Insert an 80x80 patch from a donor image (foreign content splice)."""
    out = img.copy()
    h, w = out.shape[:2]
    dh, dw = donor.shape[:2]
    out[h//4:h//4+80, w//4:w//4+80] = donor[dh//4:dh//4+80, dw//4:dw//4+80]
    return out

def atk_social_media_jpeg(img):
    """JPEG re-compression at Q=80 (typical social-media sharing pipeline)."""
    _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

# ── Main experiment ───────────────────────────────────────────────────────────
def run():
    print("\n=== Real-World Validation (Kodak Dataset) ===\n")

    # Download all images
    raw_paths = {}
    for name, url in KODAK_IMAGES.items():
        dest = os.path.join(OUT_DIR, "raw", f"{name}.png")
        if download(url, dest):
            raw_paths[name] = dest

    if not raw_paths:
        print("No images available. Check network connection.")
        return

    names = list(raw_paths.keys())
    rows  = []

    for name in names:
        print(f"\n[{name}]")
        orig = load_crop_resize(raw_paths[name])
        if orig is None:
            print(f"  Cannot load {raw_paths[name]}, skipping.")
            continue

        # Donor image for splicing (next image in cycle)
        donor_name = names[(names.index(name) + 1) % len(names)]
        donor = load_crop_resize(raw_paths[donor_name])

        # Save original and embed
        orig_path = os.path.join(OUT_DIR, f"{name}_original.png")
        wmk_path  = os.path.join(OUT_DIR, f"{name}_watermarked.png")
        cv2.imwrite(orig_path, orig)
        wm.embed(orig_path, wmk_path)
        wmk = cv2.imread(wmk_path)

        attacks = {
            "credential_forgery": lambda img: atk_credential_forgery(img),
            "copy_move":          lambda img: atk_copy_move(img),
            "splicing":           lambda img, d=donor: atk_splicing(img, d),
            "social_media_jpeg":  lambda img: atk_social_media_jpeg(img),
        }

        for atk_name, atk_fn in attacks.items():
            atk     = atk_fn(wmk)
            atk_path = os.path.join(OUT_DIR, f"{name}_{atk_name}_attacked.png")
            rec_path = os.path.join(OUT_DIR, f"{name}_{atk_name}_recovered.png")
            cv2.imwrite(atk_path, atk)
            wm.recover(atk_path, rec_path)

            rec = cv2.imread(rec_path)
            if rec is None:
                continue

            tau, eta, branch = classify_branch(atk)
            rp = psnr(orig, rec)
            rs = compute_ssim(orig, rec)
            rm = compute_msssim(orig, rec)

            print(f"  {atk_name:22s} | Br.{branch} | "
                  f"tau={tau:5.1f}% | eta={eta:.3f}% | "
                  f"PSNR={rp:6.2f} dB | SSIM={rs:.4f} | MS-SSIM={rm:.4f}")

            rows.append({
                "Image":    name,
                "Category": name,
                "Attack":   atk_name,
                "Branch":   branch,
                "tau (%)":  f"{tau:.2f}",
                "eta (%)":  f"{eta:.3f}",
                "R-PSNR":   f"{rp:.2f}",
                "SSIM":     f"{rs:.4f}",
                "MS-SSIM":  f"{rm:.4f}",
            })

    if not rows:
        print("No results generated.")
        return

    # Per-attack summary
    print("\n=== Per-Attack Summary (mean over images) ===")
    atk_names = list(dict.fromkeys(r["Attack"] for r in rows))
    summary_rows = []
    for atk in atk_names:
        subset = [r for r in rows if r["Attack"] == atk]
        psnrs  = [float(r["R-PSNR"])   for r in subset]
        ssims  = [float(r["SSIM"])      for r in subset]
        mssims = [float(r["MS-SSIM"])   for r in subset]
        branches = list(set(r["Branch"] for r in subset))

        mp, sp, cp = ci95(psnrs)
        ms_, ss_, cs = ci95(ssims)
        mm, sm, cm = ci95(mssims)

        print(f"  {atk:22s} | Br.{'/'.join(branches)} | "
              f"PSNR={mp:.2f}+/-{cp:.2f} | SSIM={ms_:.4f}+/-{cs:.4f}")

        summary_rows.append({
            "Attack":          atk,
            "Branch":          "/".join(branches),
            "R-PSNR mean":     f"{mp:.2f}",
            "R-PSNR std":      f"{sp:.2f}",
            "R-PSNR CI95":     f"{cp:.2f}",
            "SSIM mean":       f"{ms_:.4f}",
            "SSIM std":        f"{ss_:.4f}",
            "SSIM CI95":       f"{cs:.4f}",
            "MS-SSIM mean":    f"{mm:.4f}",
        })

    # Save CSVs
    detail_csv = os.path.join(OUT_DIR, "Table_RealWorld_Detail.csv")
    summary_csv = os.path.join(OUT_DIR, "Table_RealWorld_Summary.csv")

    with open(detail_csv, "w", newline="", encoding="utf-8-sig") as f:
        dw = csv.DictWriter(f, fieldnames=rows[0].keys())
        dw.writeheader(); dw.writerows(rows)

    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        sw = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        sw.writeheader(); sw.writerows(summary_rows)

    print(f"\nSaved: {detail_csv}")
    print(f"Saved: {summary_csv}")
    print("\nDone.")

if __name__ == "__main__":
    run()
