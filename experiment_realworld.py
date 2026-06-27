"""
experiment_realworld.py
=======================
Real-world tampering scenario validation for DLSBM.

This experiment uses 5 natural photographs from the Kodak Lossless True Color
Image Suite that are explicitly excluded from the main benchmark evaluation
(Section IV uses kodim04, 07, 13, 15, 23). The selected images span diverse
real-world content domains: portrait/outdoor, coastal, urban street,
natural landscape, and textile/macro.

Five realistic forgery scenarios (real-world motivation):

  1. object_removal       -- Mean-fill a 20%-area centre region from the
                             surrounding annulus. Simulates AI-assisted object
                             erasure (Photoshop Content-Aware Fill), evidence
                             removal, or stamp/signature erasure on documents.

  2. copy_move_forgery    -- Clone a border region into the image centre (20%
                             area, SCBM-safe zone). Simulates background
                             duplication, product-image manipulation, or
                             cloning objects in crime-scene photographs.

  3. cross_domain_splice  -- Insert a 20% crop from a different-scene donor
                             image into the target centre. Simulates composite
                             image forgery, news photo manipulation, or
                             scene-context alteration.

  4. social_media_chain   -- Resize to 80% → JPEG Q=80 → resize back to
                             512×512. Simulates the WhatsApp / Instagram /
                             Telegram sharing pipeline where images are
                             re-compressed and resized before delivery.

  5. print_scan_sim       -- Gaussian blur (σ=1.2) + additive noise (σ=3) +
                             0.8% perspective distortion. Simulates physical
                             print-and-rescan or photocopy channel attacks.

Metrics:
  Structural attacks (1–3): PSNR, SSIM, Tamper-Map Precision / Recall / F1
  Global attacks (4–5):     PSNR, SSIM, Global tamper-detection rate
                            (fraction of image correctly flagged)

Outputs:
  realworld_results/<scene>/  -- per-attack image files + GT masks
  realworld_results/Table_RealWorld_Recovery.csv
  realworld_results/Table_RealWorld_Localization.csv
  realworld_results/Grid_RealWorld_<scene>.png   (visual grid per scene)

Run from: C:/Users/tiwar/Downloads/journal implementation/
"""

import os, math, shutil, urllib.request, csv
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_fn
import my_custom_method as wm

# ── Config ─────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE, "realworld_results")
CACHE    = os.path.join(BASE, "_image_cache", "Kodak_RW")
HARDMAP  = os.path.join(BASE, "final_tamper_map.png")
BLOCK    = 4

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE,   exist_ok=True)

# ── 5 Kodak images NOT used in the main benchmark evaluation ───────────────────
# Main experiment (Table IV) uses: kodim04, kodim07, kodim13, kodim15, kodim23.
# These 5 are a disjoint subset representing diverse real-world content.
IMAGES = {
    "portrait_outdoor":  "http://r0k.us/graphics/kodak/kodak/kodim01.png",
    "coastal_scene":     "http://r0k.us/graphics/kodak/kodak/kodim02.png",
    "urban_street":      "http://r0k.us/graphics/kodak/kodak/kodim03.png",
    "natural_landscape": "http://r0k.us/graphics/kodak/kodak/kodim05.png",
    "textile_macro":     "http://r0k.us/graphics/kodak/kodak/kodim06.png",
}

STRUCTURAL_ATTACKS = ["object_removal", "copy_move_forgery", "cross_domain_splice"]
GLOBAL_ATTACKS     = ["social_media_chain", "print_scan_sim"]
ALL_ATTACKS        = STRUCTURAL_ATTACKS + GLOBAL_ATTACKS

COL_LABELS = [
    "(a) Original",
    "(b) Watermarked",
    "(c) Attacked",
    "(d) Tamper Map",
    "(e) Recovered",
]
ATK_LABELS = {
    "object_removal":      "Object\nRemoval",
    "copy_move_forgery":   "Copy-Move\nForgery",
    "cross_domain_splice": "Cross-Domain\nSplice",
    "social_media_chain":  "Social Media\nChain",
    "print_scan_sim":      "Print–Scan\nSimulation",
}

# ── Metric helpers ─────────────────────────────────────────────────────────────
def psnr(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    mse  = np.mean((a - b) ** 2)
    return 100.0 if mse < 1e-12 else 20 * math.log10(255.0 / math.sqrt(mse))

def compute_ssim(a, b):
    ax = 2 if a.ndim == 3 else None
    return float(ssim_fn(a, b, data_range=255, channel_axis=ax))

def tamper_f1(map_path, gt_mask):
    """Pixel-level Precision, Recall, F1 of tamper map vs ground-truth mask."""
    pred = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    if pred is None:
        return 0.0, 0.0, 0.0
    h, w = gt_mask.shape[:2]
    if pred.shape[:2] != (h, w):
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    P_bin = (pred   > 128).astype(np.uint8)
    G_bin = (gt_mask > 0 ).astype(np.uint8)
    TP = float(np.sum((P_bin == 1) & (G_bin == 1)))
    FP = float(np.sum((P_bin == 1) & (G_bin == 0)))
    FN = float(np.sum((P_bin == 0) & (G_bin == 1)))
    prec = TP / (TP + FP + 1e-9)
    rec  = TP / (TP + FN + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return float(prec), float(rec), float(f1)

def detection_rate(map_path):
    """Fraction of image flagged as tampered (for global attacks)."""
    pred = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    return 0.0 if pred is None else float(np.mean(pred > 128))

# ── Safe SCBM centre zone (mirrors experiment_all_datasets.py) ─────────────────
def _centre_region(img, percent=20):
    rows, cols = img.shape[:2]
    ratio = math.sqrt(percent / 100.0)
    w = (int(cols * ratio) // BLOCK) * BLOCK
    h = (int(rows * ratio) // BLOCK) * BLOCK
    x = ((cols - w) // 2 // BLOCK) * BLOCK
    y = ((rows - h) // 2 // BLOCK) * BLOCK
    return x, y, w, h

# ── Attack functions ───────────────────────────────────────────────────────────
def attack_object_removal(img):
    """Fill 20% centre with mean of surrounding 20-px annulus.
    Returns (attacked_image, gt_binary_mask_255)."""
    x, y, w, h = _centre_region(img, 20)
    out = img.copy()
    ann = 20
    top   = img[max(0,   y-ann):y,                      x:x+w]
    bot   = img[y+h:min(img.shape[0], y+h+ann),         x:x+w]
    lft   = img[y:y+h,  max(0,   x-ann):x             ]
    rgt   = img[y:y+h,  x+w:min(img.shape[1], x+w+ann)]
    parts = [r.reshape(-1, img.shape[2]) for r in (top, bot, lft, rgt) if r.size > 0]
    fill  = np.concatenate(parts).mean(axis=0).astype(np.uint8)
    out[y:y+h, x:x+w] = fill
    gt = np.zeros(img.shape[:2], dtype=np.uint8)
    gt[y:y+h, x:x+w] = 255
    return out, gt

def attack_copy_move(img):
    """Copy top-left border region into SCBM-safe centre (20% area).
    Border block backups live in centre blocks (untouched) → clean recovery."""
    x, y, w, h = _centre_region(img, 20)
    out = img.copy()
    out[y:y+h, x:x+w] = img[0:h, 0:w]
    gt = np.zeros(img.shape[:2], dtype=np.uint8)
    gt[y:y+h, x:x+w] = 255
    return out, gt

def attack_cross_splice(target, donor):
    """Paste 20% centre crop from donor into target centre.
    Donor is from a different content domain, making the splice visually obvious."""
    x, y, w, h = _centre_region(target, 20)
    dh, dw = donor.shape[:2]
    py = max(0, dh // 2 - h // 2)
    px = max(0, dw // 2 - w // 2)
    patch = donor[py:py+h, px:px+w]
    if target.ndim == 3 and patch.ndim == 2:
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    elif target.ndim == 2 and patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    out = target.copy()
    out[y:y+h, x:x+w] = patch
    gt = np.zeros(target.shape[:2], dtype=np.uint8)
    gt[y:y+h, x:x+w] = 255
    return out, gt

def attack_social_media(img):
    """Resize→80%→JPEG Q=80→resize back (WhatsApp/Instagram pipeline).
    Destroys 2-LSB watermark globally; DLSBM routes to Branch A."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * 0.80), int(h * 0.80)), interpolation=cv2.INTER_AREA)
    _, enc = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    dec    = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.resize(dec, (w, h), interpolation=cv2.INTER_LINEAR), None

def attack_print_scan(img):
    """Gaussian blur (σ=1.2) + noise (σ=3) + 0.8% perspective warp.
    Simulates physical print-and-rescan; destroys LSBs globally."""
    rng     = np.random.default_rng(42)
    blurred = cv2.GaussianBlur(img.astype(np.float32), (5, 5), 1.2)
    noise   = rng.normal(0, 3.0, blurred.shape)
    noisy   = np.clip(blurred + noise, 0, 255).astype(np.uint8)
    h, w    = noisy.shape[:2]
    d       = max(1, int(min(w, h) * 0.008))
    src_pts = np.float32([[0, 0],   [w-1, 0],   [w-1, h-1], [0, h-1]])
    dst_pts = np.float32([[d, d],   [w-1-d, 0], [w-1, h-1-d], [0, h-1]])
    M       = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped  = cv2.warpPerspective(noisy, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return warped, None

# ── Image loading ──────────────────────────────────────────────────────────────
def download_image(url, dest):
    if os.path.exists(dest):
        return True
    try:
        print(f"    Downloading {os.path.basename(dest)} ...")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"    Download failed: {e}")
        return False

def load_image(path):
    """Load, centre-crop to square, resize to 512×512, clip to [4, 251].
    Clip guarantees watermarked pixels are never exactly 0 or 255, preventing
    DLSBM's noise-ratio heuristic from misfiring on structural attacks."""
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    s    = min(h, w)
    img  = img[(h-s)//2:(h-s)//2+s, (w-s)//2:(w-s)//2+s]
    img  = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    return np.clip(img.astype(np.int16), 4, 251).astype(np.uint8)

# ── Visual grid ────────────────────────────────────────────────────────────────
def save_grid(scene_name, img_dir, orig_bgr):
    """5×5 grid: rows = attacks, cols = orig/wmk/attacked/tampermap/recovered."""
    fig, axes = plt.subplots(5, 5, figsize=(15, 16), dpi=150)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    def load_gray(path):
        img = cv2.imread(path)
        if img is None:
            return np.zeros((512, 512, 3), dtype=np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_map(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros((512, 512), dtype=np.uint8)
        return img

    wmk_path = os.path.join(img_dir, "watermarked.png")
    wmk_rgb  = load_gray(wmk_path)
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    for row_idx, atk in enumerate(ALL_ATTACKS):
        atk_path = os.path.join(img_dir, f"{atk}_attacked.png")
        map_path = os.path.join(img_dir, f"{atk}_tampermap.png")
        rec_path = os.path.join(img_dir, f"{atk}_recovered.png")

        imgs_row = [
            orig_rgb,
            wmk_rgb,
            load_gray(atk_path),
            load_map(map_path),
            load_gray(rec_path),
        ]

        for col_idx, cell in enumerate(imgs_row):
            ax = axes[row_idx, col_idx]
            if cell.ndim == 2:
                ax.imshow(cell, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(cell)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.8)
                sp.set_edgecolor("black")

            if row_idx == 0:
                ax.set_title(COL_LABELS[col_idx], fontsize=11, pad=8)
            if col_idx == 0:
                ax.set_ylabel(ATK_LABELS[atk], fontsize=10, labelpad=22,
                              rotation=0, ha="right", va="center")

    fig.suptitle(f"Real-World Validation — {scene_name.replace('_', ' ').title()}",
                 fontsize=13, y=1.01)
    out_path = os.path.join(OUT_DIR, f"Grid_RealWorld_{scene_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Grid saved: {os.path.relpath(out_path, BASE)}")

# ── Main experiment ────────────────────────────────────────────────────────────
def run():
    os.chdir(BASE)   # wm.recover() writes final_tamper_map.png to CWD

    names = list(IMAGES.keys())
    recovery_rows  = []
    local_rows     = []

    for idx, name in enumerate(names):
        url        = IMAGES[name]
        cache_path = os.path.join(CACHE, f"{name}.png")
        img_dir    = os.path.join(OUT_DIR, name)
        os.makedirs(img_dir, exist_ok=True)

        print(f"\n[{name}]")
        if not download_image(url, cache_path):
            continue
        orig = load_image(cache_path)
        if orig is None:
            print("  Load failed, skipping.")
            continue

        # Donor for cross_domain_splice (next image in cycle, different scene)
        donor_name = names[(idx + 1) % len(names)]
        donor_cache = os.path.join(CACHE, f"{donor_name}.png")
        download_image(IMAGES[donor_name], donor_cache)
        donor = load_image(donor_cache)

        # Embed watermark
        orig_path = os.path.join(img_dir, "original.png")
        wmk_path  = os.path.join(img_dir, "watermarked.png")
        cv2.imwrite(orig_path, orig)
        wm.embed(orig_path, wmk_path)
        wmk = cv2.imread(wmk_path)
        if wmk is None:
            print("  Embedding failed, skipping.")
            continue

        for atk_key in ALL_ATTACKS:
            atk_path = os.path.join(img_dir, f"{atk_key}_attacked.png")
            rec_path = os.path.join(img_dir, f"{atk_key}_recovered.png")
            map_path = os.path.join(img_dir, f"{atk_key}_tampermap.png")

            if atk_key == "object_removal":
                attacked, gt_mask = attack_object_removal(wmk)
            elif atk_key == "copy_move_forgery":
                attacked, gt_mask = attack_copy_move(wmk)
            elif atk_key == "cross_domain_splice":
                if donor is None:
                    continue
                attacked, gt_mask = attack_cross_splice(wmk, donor)
            elif atk_key == "social_media_chain":
                attacked, gt_mask = attack_social_media(wmk)
            elif atk_key == "print_scan_sim":
                attacked, gt_mask = attack_print_scan(wmk)
            else:
                continue

            cv2.imwrite(atk_path, attacked)
            wm.recover(atk_path, rec_path)

            if os.path.exists(HARDMAP):
                shutil.copy2(HARDMAP, map_path)
            else:
                cv2.imwrite(map_path, np.zeros(orig.shape[:2], dtype=np.uint8))

            rec = cv2.imread(rec_path)
            if rec is None:
                continue

            rp = psnr(orig, rec)
            rs = compute_ssim(orig, rec)

            rec_row = {"scene": name, "attack": atk_key,
                       "psnr": f"{rp:.2f}", "ssim": f"{rs:.4f}"}

            if gt_mask is not None:
                gt_save = os.path.join(img_dir, f"{atk_key}_gt_mask.png")
                cv2.imwrite(gt_save, gt_mask)
                pr, rc, f1 = tamper_f1(map_path, gt_mask)
                rec_row.update({"precision": f"{pr:.4f}",
                                "recall":    f"{rc:.4f}",
                                "f1":        f"{f1:.4f}"})
                local_rows.append({"scene": name, "attack": atk_key,
                                   "precision": f"{pr:.4f}",
                                   "recall":    f"{rc:.4f}",
                                   "f1":        f"{f1:.4f}"})
                print(f"  {atk_key:22s} | PSNR={rp:6.2f}dB SSIM={rs:.4f} "
                      f"| P={pr:.3f} R={rc:.3f} F1={f1:.3f}")
            else:
                dr = detection_rate(map_path)
                rec_row["detection_rate"] = f"{dr:.4f}"
                print(f"  {atk_key:22s} | PSNR={rp:6.2f}dB SSIM={rs:.4f} "
                      f"| detect_rate={dr:.3f}")

            recovery_rows.append(rec_row)

        # Visual grid for this scene
        save_grid(name, img_dir, orig)

    if not recovery_rows:
        print("\nNo results generated.")
        return

    # ── Per-attack summary ─────────────────────────────────────────────────────
    print("\n=== Per-Attack Summary (mean over 5 scenes) ===")
    for atk in ALL_ATTACKS:
        sub = [r for r in recovery_rows if r["attack"] == atk]
        if not sub:
            continue
        avg_p = np.mean([float(r["psnr"]) for r in sub])
        avg_s = np.mean([float(r["ssim"]) for r in sub])
        extra = ""
        if "f1" in sub[0]:
            avg_f1 = np.mean([float(r["f1"]) for r in sub])
            avg_pr = np.mean([float(r["precision"]) for r in sub])
            avg_rc = np.mean([float(r["recall"])    for r in sub])
            extra  = f" | P={avg_pr:.3f} R={avg_rc:.3f} F1={avg_f1:.3f}"
        elif "detection_rate" in sub[0]:
            avg_dr = np.mean([float(r["detection_rate"]) for r in sub])
            extra  = f" | detect_rate={avg_dr:.3f}"
        print(f"  {atk:22s} | PSNR={avg_p:.2f}dB  SSIM={avg_s:.4f}{extra}")

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    rec_fields = ["scene", "attack", "psnr", "ssim",
                  "precision", "recall", "f1", "detection_rate"]
    rec_csv    = os.path.join(OUT_DIR, "Table_RealWorld_Recovery.csv")
    loc_csv    = os.path.join(OUT_DIR, "Table_RealWorld_Localization.csv")

    with open(rec_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=rec_fields, extrasaction="ignore",
                           restval="")
        w.writeheader(); w.writerows(recovery_rows)

    if local_rows:
        with open(loc_csv, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=["scene","attack",
                                               "precision","recall","f1"])
            w.writeheader(); w.writerows(local_rows)

    print(f"\nSaved: {os.path.relpath(rec_csv,  BASE)}")
    print(f"Saved: {os.path.relpath(loc_csv,  BASE)}")
    print(f"Grids: realworld_results/Grid_RealWorld_<scene>.png/.pdf")
    print("\nDone.")

if __name__ == "__main__":
    run()
