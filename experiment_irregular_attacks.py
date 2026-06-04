"""
experiment_irregular_attacks.py
================================
Supplementary evaluation: tamper detection under irregular (non-rectangular,
semantically realistic) forgeries.

Regular attacks (Figure 5): geometric regions -- rectangles, uniform JPEG,
                              uniform noise.
Irregular attacks (this file): non-rectangular boundaries, multi-region
                                scatter, and semantic object duplication with
                                an Otsu-derived mask.

Attack types
------------
  1. semantic_duplicate  -- Otsu-masked copy-move; boundary follows object
                            silhouette (e.g. sailboat, jet, face, pepper)
  2. polygon_splice      -- 7-vertex irregular polygon copy-move
  3. scatter_tamper      -- 8 small non-contiguous scattered regions (~15% area)
  4. diagonal_band       -- 45-degree diagonal band copy-move

Outputs
-------
  irregular_attack_results/
    0_Watermarked/          -- watermarked images (shared across attacks)
    <attack>/Attacked/      -- attacked images
    <attack>/Recovered/     -- recovered images
    <attack>/Tamper_Maps/   -- tamper localization maps from recover()
  irregular_attack_grids/
    Grid_<image>.png/.pdf   -- N_attacks x 5 visual grid (same style as
                               varying_tamper_grids/)
  Table_Irregular_Attacks.csv
"""

import cv2
import numpy as np
import os
import glob
import math
import csv
import shutil
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from skimage.metrics import structural_similarity as _ssim

import my_custom_method as watermark_system
from generate_detection_metrics import (
    predicted_block_mask,
    groundtruth_block_mask,
    confusion,
    metrics_from_counts,
)

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
BLOCK_SIZE  = 4
INPUT_DIR   = "grayscale_normalized"
RESULTS_DIR = "irregular_attack_results"
GRID_DIR    = "irregular_attack_grids"
CSV_OUT     = "Table_Irregular_Attacks.csv"

IRREGULAR_ATTACKS = [
    ("semantic_duplicate", "Semantic\nDuplicate"),
    ("polygon_splice",     "Polygon\nSplice"),
    ("scatter_tamper",     "Scatter\nTamper"),
    ("diagonal_band",      "Diagonal\nBand"),
]

COL_LABELS = [
    "(a) Original",
    "(b) Watermarked",
    "(c) Attacked",
    "(d) Tamper Map",
    "(e) Recovered",
]

# -----------------------------------------------------------------------
# Image-specific semantic object parameters
# -----------------------------------------------------------------------
# Each entry: (src_x_frac, src_y_frac, src_w_frac, src_h_frac,
#              dst_x_frac, dst_y_frac)
# Validated: dst_x_frac + src_w_frac <= 1 and dst_y_frac + src_h_frac <= 1
# Src region chosen to cover the principal semantic object; dst is clearly
# separated so the duplication is visually unambiguous.
SEMANTIC_OBJECTS = {
    "Boat":            (0.05, 0.40, 0.38, 0.50, 0.58, 0.02),  # sailboat -> sky/upper-right
    "JetPlane":        (0.15, 0.25, 0.65, 0.55, 0.10, 0.02),  # jet body -> top sky
    "Mandril":         (0.20, 0.30, 0.60, 0.45, 0.10, 0.02),  # face center -> top
    "Peppers":         (0.02, 0.10, 0.45, 0.80, 0.50, 0.10),  # left pepper -> right side
    "Walter-Cronkite": (0.20, 0.05, 0.60, 0.70, 0.02, 0.25),  # face -> left edge
    "Lake":            (0.28, 0.63, 0.45, 0.35, 0.02, 0.05),  # terrain cluster -> water area
    "Houses":          (0.05, 0.05, 0.45, 0.45, 0.50, 0.50),  # house cluster -> other half
    "Clock":           (0.20, 0.10, 0.60, 0.80, 0.02, 0.05),  # clock face -> left
    "Chemicalplant":   (0.46, 0.50, 0.48, 0.45, 0.02, 0.02),  # main structure -> top-left
}
_DEFAULT_OBJECT = (0.10, 0.10, 0.40, 0.40, 0.52, 0.52)


# -----------------------------------------------------------------------
# Directory setup
# -----------------------------------------------------------------------
def setup_directories():
    os.makedirs(os.path.join(RESULTS_DIR, "0_Watermarked"), exist_ok=True)
    os.makedirs(GRID_DIR, exist_ok=True)
    for atk_key, _ in IRREGULAR_ATTACKS:
        for sub in ("Attacked", "Recovered", "Tamper_Maps"):
            os.makedirs(os.path.join(RESULTS_DIR, atk_key, sub), exist_ok=True)


# -----------------------------------------------------------------------
# Attack 1: Semantic Duplicate
# Copy the principal object (irregular Otsu mask) to a distinct location.
# -----------------------------------------------------------------------
def attack_semantic_duplicate(wm_img, base_name):
    h, w = wm_img.shape[:2]
    sx_f, sy_f, sw_f, sh_f, dx_f, dy_f = SEMANTIC_OBJECTS.get(
        base_name, _DEFAULT_OBJECT)

    # Pixel coords, block-aligned
    sx = (int(sx_f * w) // BLOCK_SIZE) * BLOCK_SIZE
    sy = (int(sy_f * h) // BLOCK_SIZE) * BLOCK_SIZE
    sw = (int(sw_f * w) // BLOCK_SIZE) * BLOCK_SIZE
    sh = (int(sh_f * h) // BLOCK_SIZE) * BLOCK_SIZE
    dx = (int(dx_f * w) // BLOCK_SIZE) * BLOCK_SIZE
    dy = (int(dy_f * h) // BLOCK_SIZE) * BLOCK_SIZE

    # Clamp region to fit both src and dst within image
    sw = max(BLOCK_SIZE, min(sw, w - sx, w - dx))
    sh = max(BLOCK_SIZE, min(sh, h - sy, h - dy))

    # Otsu threshold inside src region to get irregular silhouette mask
    gray = cv2.cvtColor(wm_img, cv2.COLOR_BGR2GRAY)
    src_gray = gray[sy:sy + sh, sx:sx + sw]
    _, mask = cv2.threshold(src_gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup: close small holes, smooth boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    # If >85% is foreground the region is nearly uniform -- invert to keep
    # the more informative (darker/lighter) minority as the object
    if np.mean(mask > 0) > 0.85:
        mask = cv2.bitwise_not(mask)

    # Fallback: if mask is still nearly empty, use the full rectangle
    if np.count_nonzero(mask) < 0.05 * mask.size:
        mask[:] = 255

    mask_bool = mask > 0  # (sh, sw) boolean

    attacked = wm_img.copy()
    for c in range(3):
        dst_ch = attacked[dy:dy + sh, dx:dx + sw, c].copy()
        src_ch = wm_img[sy:sy + sh, sx:sx + sw, c]
        dst_ch[mask_bool] = src_ch[mask_bool]
        attacked[dy:dy + sh, dx:dx + sw, c] = dst_ch

    return attacked, "Semantic Duplicate"


# -----------------------------------------------------------------------
# Attack 2: Polygon Splice
# An irregular 7-vertex polygon copy-move region.
# -----------------------------------------------------------------------
def attack_polygon_splice(wm_img):
    h, w = wm_img.shape[:2]

    # Polygon centre: right-centre of image
    cx = int(w * 0.65)
    cy = int(h * 0.35)
    r  = int(min(h, w) * 0.18)

    # 7 vertices at irregular angles/radii so it is clearly non-rectangular
    angles_deg = [0, 51, 110, 170, 230, 285, 335]
    scale      = [1.00, 0.75, 1.10, 0.80, 1.05, 0.70, 0.95]
    pts = np.array([
        [np.clip(int(cx + scale[i] * r * math.cos(math.radians(a))), 0, w - 1),
         np.clip(int(cy + scale[i] * r * math.sin(math.radians(a))), 0, h - 1)]
        for i, a in enumerate(angles_deg)
    ], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # Source content: shift from upper-left quadrant (non-overlapping)
    shift_x = int(w * 0.38)
    shift_y = int(h * 0.38)

    dst_y, dst_x = np.where(mask > 0)
    src_y_arr = np.clip(dst_y - shift_y, 0, h - 1)
    src_x_arr = np.clip(dst_x - shift_x, 0, w - 1)

    attacked = wm_img.copy()
    attacked[dst_y, dst_x] = wm_img[src_y_arr, src_x_arr]

    return attacked, "Polygon Splice"


# -----------------------------------------------------------------------
# Attack 3: Scatter Tamper
# 8 small non-contiguous copy-move regions scattered across the image.
# -----------------------------------------------------------------------
def attack_scatter_tamper(wm_img, n_regions=8, seed=42):
    h, w   = wm_img.shape[:2]
    rng    = np.random.RandomState(seed)

    # Each region ~5% of the shorter side, block-aligned
    rw = max(BLOCK_SIZE,
             (int(min(h, w) * 0.05) // BLOCK_SIZE) * BLOCK_SIZE)
    rh = rw

    attacked = wm_img.copy()
    placed   = []  # list of (dx, dy) already placed

    for _ in range(n_regions):
        # Try up to 60 times to find a non-overlapping placement
        for _ in range(60):
            dx = (rng.randint(0, max(1, w - rw)) // BLOCK_SIZE) * BLOCK_SIZE
            dy = (rng.randint(0, max(1, h - rh)) // BLOCK_SIZE) * BLOCK_SIZE
            sx = (rng.randint(0, max(1, w - rw)) // BLOCK_SIZE) * BLOCK_SIZE
            sy = (rng.randint(0, max(1, h - rh)) // BLOCK_SIZE) * BLOCK_SIZE

            src_dst_far = abs(dx - sx) >= rw * 2 or abs(dy - sy) >= rh * 2
            no_overlap  = all(abs(dx - px) >= rw or abs(dy - py) >= rh
                              for px, py in placed)
            if src_dst_far and no_overlap:
                break

        placed.append((dx, dy))
        attacked[dy:dy + rh, dx:dx + rw] = wm_img[sy:sy + rh, sx:sx + rw]

    return attacked, f"Scatter Tamper ({n_regions} regions)"


# -----------------------------------------------------------------------
# Attack 4: Diagonal Band
# A 45-degree diagonal band copy-move (non-axis-aligned boundary).
# -----------------------------------------------------------------------
def attack_diagonal_band(wm_img, band_frac=0.08):
    h, w = wm_img.shape[:2]

    # Perpendicular distance from the main diagonal (top-left to bottom-right)
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    diag_len  = math.sqrt(float(h) ** 2 + float(w) ** 2)
    dist      = np.abs(Y * w - X * h) / diag_len
    band_px   = min(h, w) * band_frac
    mask      = dist < band_px

    # Shift source perpendicular to the diagonal (toward lower-right)
    perp_shift = int(band_px * 2.5)
    shift      = (max(perp_shift, BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
    step       = int(shift / math.sqrt(2.0))

    dst_y, dst_x = np.where(mask)
    src_y_arr = np.clip(dst_y + step, 0, h - 1)
    src_x_arr = np.clip(dst_x + step, 0, w - 1)

    attacked = wm_img.copy()
    attacked[dst_y, dst_x] = wm_img[src_y_arr, src_x_arr]

    return attacked, "Diagonal Band"


# -----------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------
def apply_attack(atk_key, wm_img, base_name):
    if atk_key == "semantic_duplicate":
        return attack_semantic_duplicate(wm_img, base_name)
    if atk_key == "polygon_splice":
        return attack_polygon_splice(wm_img)
    if atk_key == "scatter_tamper":
        return attack_scatter_tamper(wm_img)
    if atk_key == "diagonal_band":
        return attack_diagonal_band(wm_img)
    return None, None


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------
def calculate_psnr(img1, img2):
    """PSNR (dB) between two uint8 grayscale images."""
    if img1 is None or img2 is None:
        return float("nan")
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    """SSIM between two uint8 grayscale images."""
    if img1 is None or img2 is None:
        return float("nan")
    return float(_ssim(img1, img2, data_range=255))


def evaluate_pair(wm_img, atk_img):
    gt   = groundtruth_block_mask(wm_img, atk_img)
    pred = predicted_block_mask(atk_img)
    tp, fp, tn, fn = confusion(gt, pred)
    return metrics_from_counts(tp, fp, tn, fn)


def _fmt(v):
    return ("N/A" if v is None or
            (isinstance(v, float) and math.isnan(v))
            else f"{v:.2f}")

def _fmt4(v):
    return ("N/A" if v is None or
            (isinstance(v, float) and math.isnan(v))
            else f"{v:.4f}")


# -----------------------------------------------------------------------
# Pipeline: embed once, then attack / recover / evaluate per attack
# -----------------------------------------------------------------------
def run_pipeline(file_path, base_name):
    png_filename = f"{base_name}.png"
    wm_path = os.path.join(RESULTS_DIR, "0_Watermarked", png_filename)

    # Embed (skip if already done)
    if not os.path.exists(wm_path):
        print(f"  [embed] {base_name}")
        if not watermark_system.embed(file_path, wm_path):
            print("  ERROR: embed failed")
            return {}

    wm_img = cv2.imread(wm_path)
    if wm_img is None:
        print(f"  ERROR: cannot read {wm_path}")
        return {}

    # --- WPSNR / WSSIM: original vs watermarked (same for all attacks) ---
    orig_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    wm_gray   = cv2.cvtColor(wm_img, cv2.COLOR_BGR2GRAY)
    wpsnr = calculate_psnr(orig_gray, wm_gray)
    wssim = calculate_ssim(orig_gray, wm_gray)
    print(f"  [watermark quality]  WPSNR={wpsnr:.2f} dB  WSSIM={wssim:.4f}")

    results = {}
    for atk_key, _ in IRREGULAR_ATTACKS:
        print(f"  [{atk_key}] ... ", end="", flush=True)

        atk_path = os.path.join(RESULTS_DIR, atk_key, "Attacked",    png_filename)
        rec_path = os.path.join(RESULTS_DIR, atk_key, "Recovered",   png_filename)
        map_path = os.path.join(RESULTS_DIR, atk_key, "Tamper_Maps", png_filename)

        # Apply attack
        atk_img, _ = apply_attack(atk_key, wm_img, base_name)
        if atk_img is None:
            print("FAILED (attack returned None)")
            continue
        cv2.imwrite(atk_path, atk_img)

        # Recover and capture tamper map immediately
        watermark_system.recover(atk_path, rec_path)
        src_map = "final_tamper_map.png"
        if os.path.exists(src_map):
            shutil.copy2(src_map, map_path)
        else:
            blank = np.zeros(wm_img.shape[:2], dtype=np.uint8)
            cv2.imwrite(map_path, blank)

        # --- RPSNR / RSSIM: original vs recovered ---
        rec_gray = cv2.imread(rec_path, cv2.IMREAD_GRAYSCALE)
        rpsnr = calculate_psnr(orig_gray, rec_gray)
        rssim = calculate_ssim(orig_gray, rec_gray)

        # Detection metrics
        m = evaluate_pair(wm_img, atk_img)
        m["wpsnr"] = wpsnr
        m["wssim"] = wssim
        m["rpsnr"] = rpsnr
        m["rssim"] = rssim
        results[atk_key] = m

        print(f"rate={_fmt(m['rate'])}%  TPR={_fmt(m['tpr'])}%  "
              f"FPR={_fmt(m['fpr'])}%  Prec={_fmt(m['precision'])}%  "
              f"Acc={_fmt(m['accuracy'])}%  "
              f"RPSNR={rpsnr:.2f}dB  RSSIM={rssim:.4f}")

    return results


# -----------------------------------------------------------------------
# Grid: N_attacks x 5 panel for one image
# Matches generate_varying_tamper_grids.py style exactly.
# -----------------------------------------------------------------------
def create_grid(base_name):
    png_filename = f"{base_name}.png"

    # Locate original source image
    orig_path = None
    for ext in (".tiff", ".png", ".jpg", ".jpeg"):
        cand = os.path.join(INPUT_DIR, f"{base_name}{ext}")
        if os.path.exists(cand):
            orig_path = cand
            break

    wm_path  = os.path.join(RESULTS_DIR, "0_Watermarked", png_filename)
    orig_img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE) if orig_path else None
    wm_img   = cv2.imread(wm_path,   cv2.IMREAD_GRAYSCALE)

    n_rows = len(IRREGULAR_ATTACKS)
    fig, axes = plt.subplots(nrows=n_rows, ncols=5,
                             figsize=(15, n_rows * 3), dpi=300)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for row, (atk_key, row_label) in enumerate(IRREGULAR_ATTACKS):
        base_dir = os.path.join(RESULTS_DIR, atk_key)
        atk_img  = cv2.imread(os.path.join(base_dir, "Attacked",    png_filename),
                              cv2.IMREAD_GRAYSCALE)
        map_img  = cv2.imread(os.path.join(base_dir, "Tamper_Maps", png_filename),
                              cv2.IMREAD_GRAYSCALE)
        rec_img  = cv2.imread(os.path.join(base_dir, "Recovered",   png_filename),
                              cv2.IMREAD_GRAYSCALE)

        for col, img in enumerate([orig_img, wm_img, atk_img, map_img, rec_img]):
            ax = axes[row, col]

            if img is not None:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            else:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)

            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
                spine.set_edgecolor("black")

            if row == 0:
                ax.set_title(COL_LABELS[col], fontsize=14, pad=10)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=12, labelpad=25,
                              rotation=0, ha="right", va="center")

    out_base = os.path.join(GRID_DIR, f"Grid_{base_name}")
    plt.savefig(out_base + ".pdf", dpi=300, bbox_inches="tight")
    plt.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [grid] {out_base}.png/.pdf")


# -----------------------------------------------------------------------
# CSV export
# -----------------------------------------------------------------------
def write_csv(all_results, image_names):
    atk_keys   = [k for k, _ in IRREGULAR_ATTACKS]
    atk_accum  = {k: [] for k in atk_keys}

    METRIC_KEYS = ("rate", "tpr", "fpr", "precision", "accuracy",
                   "wpsnr", "wssim", "rpsnr", "rssim")

    with open(CSV_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Attack", "Image",
                    "Tampering Rate (%)", "TPR (%)", "FPR (%)",
                    "Precision (%)", "Accuracy (%)",
                    "WPSNR (dB)", "WSSIM", "RPSNR (dB)", "RSSIM"])

        prev_atk = None
        for atk_key in atk_keys:
            for base_name in image_names:
                m = all_results.get(base_name, {}).get(atk_key)
                if m is None:
                    continue
                atk_accum[atk_key].append(m)
                w.writerow([atk_key if atk_key != prev_atk else "",
                            base_name,
                            _fmt(m["rate"]),      _fmt(m["tpr"]),
                            _fmt(m["fpr"]),       _fmt(m["precision"]),
                            _fmt(m["accuracy"]),
                            _fmt(m["wpsnr"]),     _fmt4(m["wssim"]),
                            _fmt(m["rpsnr"]),     _fmt4(m["rssim"])])
                prev_atk = atk_key
            # Per-attack average
            rows = atk_accum[atk_key]
            if rows:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    avg = {k: float(np.nanmean([r[k] for r in rows]))
                           for k in METRIC_KEYS}
                w.writerow(["", "AVERAGE",
                            _fmt(avg["rate"]),    _fmt(avg["tpr"]),
                            _fmt(avg["fpr"]),     _fmt(avg["precision"]),
                            _fmt(avg["accuracy"]),
                            _fmt(avg["wpsnr"]),   _fmt4(avg["wssim"]),
                            _fmt(avg["rpsnr"]),   _fmt4(avg["rssim"])])
            w.writerow([])  # blank separator between attack groups

    print(f"\nCSV written: {CSV_OUT}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  Irregular Attacks Experiment")
    print("=" * 70)

    setup_directories()

    source_files = sorted(
        f for ext in ("*.tiff", "*.png", "*.jpg", "*.jpeg")
        for f in glob.glob(os.path.join(INPUT_DIR, ext))
    )
    if not source_files:
        print(f"ERROR: no images found in '{INPUT_DIR}'")
        return

    all_results     = {}
    processed_names = []

    for file_path in source_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\n{'=' * 60}\n  IMAGE: {base_name}\n{'=' * 60}")

        results = run_pipeline(file_path, base_name)
        all_results[base_name] = results
        processed_names.append(base_name)
        create_grid(base_name)

    write_csv(all_results, processed_names)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY  (averaged over all images)")
    print("=" * 70)
    ALL_KEYS = ("rate", "tpr", "fpr", "precision", "accuracy",
                "wpsnr", "wssim", "rpsnr", "rssim")
    atk_keys = [k for k, _ in IRREGULAR_ATTACKS]
    for atk_key in atk_keys:
        rows = [all_results.get(n, {}).get(atk_key)
                for n in processed_names]
        rows = [r for r in rows if r is not None]
        if not rows:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            avg = {k: float(np.nanmean([r[k] for r in rows]))
                   for k in ALL_KEYS}
        print(f"  {atk_key:22s}  "
              f"rate={_fmt(avg['rate']):>6}%  "
              f"TPR={_fmt(avg['tpr']):>6}%  "
              f"FPR={_fmt(avg['fpr']):>6}%  "
              f"Prec={_fmt(avg['precision']):>6}%  "
              f"Acc={_fmt(avg['accuracy']):>6}%  "
              f"WPSNR={_fmt(avg['wpsnr']):>6}dB  "
              f"WSSIM={_fmt4(avg['wssim'])}  "
              f"RPSNR={_fmt(avg['rpsnr']):>6}dB  "
              f"RSSIM={_fmt4(avg['rssim'])}")

    print(f"\nGrids  : {os.path.abspath(GRID_DIR)}/")
    print(f"Results: {os.path.abspath(RESULTS_DIR)}/")


if __name__ == "__main__":
    main()
