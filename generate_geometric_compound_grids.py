"""
generate_geometric_compound_grids.py
======================================
Creates visual result grids matching the style of irregular_attack_grids/.

Grid format (one PNG + PDF per USC-SIPI image):
  Rows    : one per attack scenario
  Columns : (a) Original  |  (b) Attacked  |  (c) Tamper Map  |  (d) Recovered

  - Tamper Map: block-level detection heat-map (tampered blocks = red overlay)
  - Recovered cell annotated with R-PSNR (dB) and SSIM in bottom-left corner

Outputs
-------
  geometric_compound_grids/geometric/Grid_<image>.png/.pdf   (13 rows × 4 cols)
  geometric_compound_grids/compound/ Grid_<image>.png/.pdf   ( 9 rows × 4 cols)

Run from:  C:/Users/tiwar/Downloads/journal implementation/
"""

import cv2
import os
import glob
import hashlib
import math
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from skimage.metrics import structural_similarity as _ssim

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_DIR   = "grayscale_normalized"
RESULTS_DIR = "geometric_compound_results"
GRID_DIR    = "geometric_compound_grids"
BLOCK_SIZE  = 4

# ── Attack display labels (plain text for matplotlib) ─────────────────────────
GEOMETRIC_LABELS = [
    "Rotation 5°",
    "Rotation 15°",
    "Rotation 30°",
    "Rotation 45°",
    "Rotation 90°",
    "Scaling ×0.50",
    "Scaling ×0.75",
    "Scaling ×1.25",
    "Scaling ×1.50",
    "Horizontal Flip",
    "Vertical Flip",
    "Rot 15° + Scale ×0.90",
    "Rot 30° + Scale ×0.80",
]

COMPOUND_LABELS = [
    "JPEG(Q=90) + S&P(3%)",
    "JPEG(Q=70) + S&P(5%)",
    "Crop(30%) + JPEG(Q=70)",
    "Crop(30%) + S&P(5%)",
    "Content Removal\n+ JPEG(Q=90)",
    "Content Removal\n+ S&P(5%)",
    "Copy-Move\n+ JPEG(Q=70)",
    "Copy-Move + S&P(5%)",
    "S&P(5%) → Crop(20%)",
]

COMPOUND_BRANCHES = ["B", "B", "B", "B", "B", "B", "A", "B", "B"]

COL_LABELS = [
    "(a) Original",
    "(b) Attacked",
    "(c) Tamper Map",
    "(d) Recovered",
]

# ── Metric helpers ────────────────────────────────────────────────────────────
def psnr(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    mse = np.mean((a - b) ** 2)
    return 100.0 if mse < 1e-12 else 20 * math.log10(255.0 / math.sqrt(mse))

def ssim_val(a, b):
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    ax = 2 if a.ndim == 3 else None
    return float(_ssim(a, b, data_range=255, channel_axis=ax))

# ── Tamper map builder ────────────────────────────────────────────────────────
def _loc_hash(flat_block, idx):
    h = hashlib.md5(
        flat_block.tobytes() + int(idx).to_bytes(4, "big")
    ).hexdigest()
    return f"{int(h[:3], 16):012b}"

def build_tamper_map(img_bgr):
    """
    Returns an H×W×3 uint8 colour image:
      - Tampered blocks  → red  overlay on the (greyed) original
      - Intact blocks    → grey version of original
    """
    img   = img_bgr.copy()
    h, w  = img.shape[:2]
    h     = (h // BLOCK_SIZE) * BLOCK_SIZE
    w     = (w // BLOCK_SIZE) * BLOCK_SIZE
    img   = img[:h, :w]

    # greyscale base for visualisation
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis   = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.uint8)

    # per-channel tamper flags: block is tampered if ANY channel fails
    bh = h // BLOCK_SIZE
    bw = w // BLOCK_SIZE
    flagged = np.zeros((bh, bw), dtype=bool)

    for ch in range(3):
        channel = img[:, :, ch]
        idx     = 0
        for bi in range(bh):
            for bj in range(bw):
                i, j  = bi * BLOCK_SIZE, bj * BLOCK_SIZE
                blk   = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
                flat  = blk.flatten()
                bits  = "".join(str((v >> b) & 1) for v in flat for b in [0, 1])
                clean = (blk & 0xFC).flatten()
                exp   = _loc_hash(clean, idx)
                if exp != bits[:12]:
                    flagged[bi, bj] = True
                idx += 1

    # paint tampered blocks red
    red_mask = np.zeros((h, w), dtype=bool)
    for bi in range(bh):
        for bj in range(bw):
            if flagged[bi, bj]:
                i, j = bi * BLOCK_SIZE, bj * BLOCK_SIZE
                red_mask[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = True

    vis[red_mask, 0] = 30    # B  (OpenCV BGR: channel 0 = Blue)
    vis[red_mask, 1] = 30    # G
    vis[red_mask, 2] = 220   # R

    return vis

# ── Grid builder ──────────────────────────────────────────────────────────────
def _to_display(img_bgr):
    """Convert BGR → RGB or keep grayscale for imshow."""
    if img_bgr is None:
        return None
    if img_bgr.ndim == 3:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr

def _annotate(ax, text, color="white"):
    """Small PSNR/SSIM label in bottom-left corner of a cell."""
    ax.text(
        0.03, 0.03, text,
        transform=ax.transAxes,
        fontsize=7.5, color=color,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, lw=0),
    )

def create_grid(base_name, attack_labels, atk_subdir, n_attacks,
                branch_labels=None, out_subdir="geometric"):
    """
    Build one grid PNG + PDF for `base_name` image.

    Parameters
    ----------
    base_name     : e.g. "Boat"
    attack_labels : list of row label strings
    atk_subdir    : "geometric" or "compound"
    n_attacks     : number of attack rows
    branch_labels : optional list of branch strings (None for geometric)
    out_subdir    : sub-folder inside GRID_DIR
    """
    png_name = f"{base_name}.png"

    # ── load original ──────────────────────────────────────────────────────
    orig_path = None
    for ext in (".tiff", ".png", ".jpg", ".jpeg"):
        c = os.path.join(INPUT_DIR, f"{base_name}{ext}")
        if os.path.exists(c):
            orig_path = c
            break
    orig_bgr = cv2.imread(orig_path) if orig_path else None

    n_rows = n_attacks
    n_cols = 4
    row_h  = 3.2          # inches per row
    fig_w  = 16           # total width inches

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(fig_w, n_rows * row_h),
        dpi=150,
    )
    plt.subplots_adjust(wspace=0.04, hspace=0.08)

    for row in range(n_rows):
        atk_path = os.path.join(
            RESULTS_DIR, atk_subdir, "attacked",
            f"{base_name}_{row:02d}.png"
        )
        rec_path = os.path.join(
            RESULTS_DIR, atk_subdir, "recovered",
            f"{base_name}_{row:02d}.png"
        )
        atk_bgr = cv2.imread(atk_path)
        rec_bgr = cv2.imread(rec_path)

        # tamper map from attacked image
        tmap_rgb = _to_display(build_tamper_map(atk_bgr)) if atk_bgr is not None else None

        # per-image PSNR / SSIM
        if orig_bgr is not None and rec_bgr is not None:
            rp = psnr(orig_bgr, rec_bgr)
            rs = ssim_val(orig_bgr, rec_bgr)
            ann = f"PSNR {rp:.1f} dB\nSSIM {rs:.4f}"
        else:
            ann = "N/A"

        images = [
            _to_display(orig_bgr),
            _to_display(atk_bgr),
            tmap_rgb,
            _to_display(rec_bgr),
        ]

        for col, img in enumerate(images):
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_xticks([])
            ax.set_yticks([])

            if img is not None:
                ax.imshow(img, vmin=0, vmax=255)
            else:
                ax.set_facecolor("#222222")
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", color="white", fontsize=9)

            # black border around each cell
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.8)
                sp.set_edgecolor("black")

            # column header (top row only)
            if row == 0:
                ax.set_title(COL_LABELS[col], fontsize=11, pad=6, fontweight="bold")

            # row label (first column only)
            if col == 0:
                row_txt = attack_labels[row]
                if branch_labels is not None:
                    row_txt += f"\n[Br. {branch_labels[row]}]"
                ax.set_ylabel(
                    row_txt, fontsize=8.5, labelpad=6,
                    rotation=0, ha="right", va="center",
                    multialignment="right",
                )

            # PSNR/SSIM annotation on recovered image
            if col == 3 and orig_bgr is not None and rec_bgr is not None:
                _annotate(ax, ann)

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir  = os.path.join(GRID_DIR, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.join(out_dir, f"Grid_{base_name}")
    fig.savefig(out_base + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  {out_base}.png")

# ── Main ──────────────────────────────────────────────────────────────────────
def get_image_bases():
    bases = []
    for ext in ("*.tiff", "*.png", "*.jpg", "*.jpeg"):
        for p in glob.glob(os.path.join(INPUT_DIR, ext)):
            bases.append(os.path.splitext(os.path.basename(p))[0])
    return sorted(set(bases))

def main():
    bases = get_image_bases()
    if not bases:
        print(f"ERROR: no images found in '{INPUT_DIR}/'"); return

    print("=" * 60)
    print("  Generating geometric attack grids ...")
    print("=" * 60)
    for base in bases:
        print(f"\n[{base}]")
        create_grid(
            base_name     = base,
            attack_labels = GEOMETRIC_LABELS,
            atk_subdir    = "geometric",
            n_attacks     = len(GEOMETRIC_LABELS),
            branch_labels = None,          # all Branch A — noted in title
            out_subdir    = "geometric",
        )

    print("\n" + "=" * 60)
    print("  Generating compound attack grids ...")
    print("=" * 60)
    for base in bases:
        print(f"\n[{base}]")
        create_grid(
            base_name     = base,
            attack_labels = COMPOUND_LABELS,
            atk_subdir    = "compound",
            n_attacks     = len(COMPOUND_LABELS),
            branch_labels = COMPOUND_BRANCHES,
            out_subdir    = "compound",
        )

    print(f"\nAll grids saved in '{GRID_DIR}/'")
    print(f"  {GRID_DIR}/geometric/  — 9 images x 13 attacks")
    print(f"  {GRID_DIR}/compound/   — 9 images x  9 attacks")

if __name__ == "__main__":
    main()
