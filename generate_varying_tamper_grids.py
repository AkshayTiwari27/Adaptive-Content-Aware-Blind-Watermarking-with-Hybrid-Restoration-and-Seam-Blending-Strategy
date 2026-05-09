"""
generate_varying_tamper_grids.py
================================
Generates a publication-quality visual grid (PDF + high-quality PNG) for
every image at each tampering percentage (10%, 20%, 30%, 40%, 50%).

Grid layout (identical to visual_grids/Grid_Lake.png):
    Rows  -> 6 attacks (Content Removal, Copy-Move, Splicing,
                         JPEG Compression, Noise, Cropping)
    Cols  -> 5 stages  (Original | Watermarked | Attacked |
                         Tamper Map | Recovered)

Attack-parameter mapping for percentage levels:
    Content Removal / Copy-Move / Splicing / Cropping
        -> area-based percentage (10-50%)
    JPEG Compression
        -> Q = 90, 80, 70, 60, 50  for 10-50% respectively
    Salt & Pepper Noise
        -> density = 0.01, 0.03, 0.05, 0.07, 0.09 for 10-50%

Output:
    varying_tamper_grids/
        10pct/ -> Grid_10pct_<image>.pdf/png
        20pct/ -> ...
        ...

Usage:
    python generate_varying_tamper_grids.py
    python generate_varying_tamper_grids.py --image Lake --pct 10 20
    python generate_varying_tamper_grids.py --force     # re-run attacks
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import glob
import math
import shutil
import argparse
import sys

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# ----------------------------------------------------------------
#  Import project modules
# ----------------------------------------------------------------
import my_custom_method as watermark_system
import attack_image as attacker

# ----------------------------------------------------------------
#  Configuration
# ----------------------------------------------------------------
INPUT_DIR   = "grayscale_normalized"
RESULTS_DIR = "varying_tamper_results"
OUTPUT_DIR  = "varying_tamper_grids"

PERCENTAGES = [10, 20, 30, 40, 50]

# All 6 attacks in the paper order (same as visual_grids)
ATTACKS = [
    "content_removal",
    "copy_move",
    "splicing",
    "jpeg_compression",
    "noise",
    "cropping",
]

# JPEG quality per percentage level (10% -> Q90, ..., 50% -> Q50)
JPEG_Q_MAP = {10: 90, 20: 80, 30: 70, 40: 60, 50: 50}

# Noise density per percentage level (from Table 9 in the paper)
NOISE_DENSITY_MAP = {10: 0.01, 20: 0.03, 30: 0.05, 40: 0.07, 50: 0.09}

# Row labels -- will be dynamically updated with Q / density info
BASE_ATTACK_LABELS = {
    "content_removal":  "Content Removal",
    "copy_move":        "Copy Move",
    "splicing":         "Splicing",
    "jpeg_compression": "JPEG Compression",
    "noise":            "Noise",
    "cropping":         "Cropping",
}

COL_LABELS = [
    "(a) Original",
    "(b) Watermarked",
    "(c) Attacked",
    "(d) Tamper Map",
    "(e) Recovered",
]


# ----------------------------------------------------------------
#  Helpers (mirrored from varying_tamper_test.py)
# ----------------------------------------------------------------
def get_attack_dimensions(rows, cols, percent):
    ratio  = math.sqrt(percent / 100.0)
    side_w = int(cols * ratio)
    side_h = int(rows * ratio)
    return side_w, side_h


def apply_percent_attack(attack_name, image, percent):
    """Apply a single attack at the given tampering percentage."""
    rows, cols = image.shape[:2]

    # --- Area-based structural attacks ---
    if attack_name == "cropping":
        return attacker.attack_cropping(image, percent=percent)

    if attack_name == "jpeg_compression":
        q = JPEG_Q_MAP[percent]
        return attacker.attack_jpeg_compression(image, quality=q)

    if attack_name == "noise":
        d = NOISE_DENSITY_MAP[percent]
        return attacker.attack_salt_and_pepper(image, amount=d)

    # Content Removal / Copy-Move / Splicing  -> area % based
    w, h = get_attack_dimensions(rows, cols, percent)
    w = (w // 4) * 4
    h = (h // 4) * 4

    if attack_name == "content_removal":
        x = ((cols - w) // 2 // 4) * 4
        y = ((rows - h) // 2 // 4) * 4
        return attacker.attack_content_removal(image, x=x, y=y, w=w, h=h)

    elif attack_name == "copy_move":
        src_x, src_y = 0, 0
        dst_x = ((cols - w) // 2 // 4) * 4
        dst_y = ((rows - h) // 2 // 4) * 4
        attacked = image.copy()
        src_region = attacked[src_y:src_y+h, src_x:src_x+w].copy()
        attacked[dst_y:dst_y+h, dst_x:dst_x+w] = src_region
        return attacked, "Copy-Move"

    elif attack_name == "splicing":
        x = ((cols - w) // 2 // 4) * 4
        y = ((rows - h) // 2 // 4) * 4
        return attacker.attack_political_splicing(image, x=x, y=y, w=w, h=h)

    return None, None


# ----------------------------------------------------------------
#  Directory key for each attack+pct combo
# ----------------------------------------------------------------
def _dir_key(atk, pct):
    """Return the subfolder name under RESULTS_DIR for a given attack+pct."""
    if atk == "jpeg_compression":
        q = JPEG_Q_MAP[pct]
        return f"jpeg_Q{q}"
    elif atk == "noise":
        d = NOISE_DENSITY_MAP[pct]
        return f"noise_{int(d*100)}pct"
    else:
        return f"{atk}_{pct}pct"


# ----------------------------------------------------------------
#  Directory setup
# ----------------------------------------------------------------
def setup_directories():
    os.makedirs(os.path.join(RESULTS_DIR, "0_Watermarked"), exist_ok=True)
    for pct in PERCENTAGES:
        os.makedirs(os.path.join(OUTPUT_DIR, f"{pct}pct"), exist_ok=True)
        for atk in ATTACKS:
            base = os.path.join(RESULTS_DIR, _dir_key(atk, pct))
            os.makedirs(os.path.join(base, "Attacked"),    exist_ok=True)
            os.makedirs(os.path.join(base, "Tamper_Maps"), exist_ok=True)
            os.makedirs(os.path.join(base, "Recovered"),   exist_ok=True)


# ----------------------------------------------------------------
#  Run (or re-use) attacks for ONE image at ONE percentage
# ----------------------------------------------------------------
def run_attacks_for_image_pct(file_path, base_name, pct, force=False):
    png_filename = f"{base_name}.png"
    wm_save_path = os.path.join(RESULTS_DIR, "0_Watermarked", png_filename)

    # 1. Embed watermark (skip if already done)
    if force or not os.path.exists(wm_save_path):
        print(f"  [embed] {base_name}")
        if not watermark_system.embed(file_path, wm_save_path):
            print(f"  ERROR: embed failed for {base_name}")
            return False

    wm_img = cv2.imread(wm_save_path)
    if wm_img is None:
        print(f"  ERROR: cannot read watermarked image {wm_save_path}")
        return False

    # 2. Attack + Recover for every attack type
    for atk in ATTACKS:
        dk       = _dir_key(atk, pct)
        base_dir = os.path.join(RESULTS_DIR, dk)
        atk_path = os.path.join(base_dir, "Attacked",    png_filename)
        rec_path = os.path.join(base_dir, "Recovered",   png_filename)
        map_path = os.path.join(base_dir, "Tamper_Maps", png_filename)

        # Skip if all outputs exist and not forced
        if not force and os.path.exists(atk_path) and \
                         os.path.exists(rec_path) and \
                         os.path.exists(map_path):
            continue

        print(f"  [attack] {atk} @ {pct}%  (dir: {dk})")
        attacked_img, _ = apply_percent_attack(atk, wm_img, pct)
        if attacked_img is None:
            print(f"    WARNING: attack returned None for {atk} @ {pct}%")
            continue

        cv2.imwrite(atk_path, attacked_img)

        print(f"  [recover] {atk} @ {pct}%")
        watermark_system.recover(atk_path, rec_path)

        # Copy tamper map before next recover() overwrites it
        hardcoded_map = "final_tamper_map.png"
        if os.path.exists(hardcoded_map):
            shutil.copy2(hardcoded_map, map_path)
        else:
            h, w = wm_img.shape[:2]
            blank = np.zeros((h, w), dtype=np.uint8)
            cv2.imwrite(map_path, blank)

    return True


# ----------------------------------------------------------------
#  Build the row label for a given attack + percentage
# ----------------------------------------------------------------
def get_row_label(atk, pct):
    """Return label like 'JPEG Compression\n(Q=70)' or 'Noise\n(d=0.05)'."""
    base = BASE_ATTACK_LABELS[atk]
    if atk == "jpeg_compression":
        q = JPEG_Q_MAP[pct]
        return f"{base}\n(Q={q})"
    elif atk == "noise":
        d = NOISE_DENSITY_MAP[pct]
        return f"{base}\n(d={d})"
    else:
        return base


# ----------------------------------------------------------------
#  Build grid for ONE image at ONE percentage
#  Style: identical to visual_grids/Grid_Lake.png
# ----------------------------------------------------------------
def create_grid(base_name, pct):
    png_filename = f"{base_name}.png"

    # Find original image
    orig_path = None
    for ext in [".tiff", ".png", ".jpg", ".jpeg"]:
        candidate = os.path.join(INPUT_DIR, f"{base_name}{ext}")
        if os.path.exists(candidate):
            orig_path = candidate
            break

    wm_path = os.path.join(RESULTS_DIR, "0_Watermarked", png_filename)

    orig_img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE) if orig_path else None
    wm_img   = cv2.imread(wm_path,   cv2.IMREAD_GRAYSCALE)

    n_rows = len(ATTACKS)   # 6
    n_cols = 5

    # --- Figure dimensions matching Grid_Lake style: 6 rows x 5 cols ---
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=(15, 18), dpi=300)

    # Spacing matching the original generate_visual_grid.py
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # --- Fill each cell ---
    for row_idx, atk in enumerate(ATTACKS):
        dk = _dir_key(atk, pct)
        base_dir = os.path.join(RESULTS_DIR, dk)

        atk_path = os.path.join(base_dir, "Attacked",    png_filename)
        map_path = os.path.join(base_dir, "Tamper_Maps", png_filename)
        rec_path = os.path.join(base_dir, "Recovered",   png_filename)

        atk_img = cv2.imread(atk_path, cv2.IMREAD_GRAYSCALE)
        map_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        rec_img = cv2.imread(rec_path, cv2.IMREAD_GRAYSCALE)

        images = [orig_img, wm_img, atk_img, map_img, rec_img]

        for col_idx, img in enumerate(images):
            ax = axes[row_idx, col_idx]

            if img is not None:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            else:
                ax.text(0.5, 0.5, "Missing\nImage",
                        ha="center", va="center")

            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # --- Black border around every cell (matching Grid_Lake) ---
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
                spine.set_edgecolor("black")

            # Column headers (top row only)
            if row_idx == 0:
                ax.set_title(COL_LABELS[col_idx], fontsize=14, pad=10)

            # Row labels (first column only)
            if col_idx == 0:
                ax.set_ylabel(
                    get_row_label(atk, pct),
                    fontsize=12, labelpad=25,
                    rotation=0, ha="right", va="center",
                )

    # --- Save ---
    out_base = os.path.join(OUTPUT_DIR, f"{pct}pct",
                            f"Grid_{pct}pct_{base_name}")
    pdf_path = out_base + ".pdf"
    png_path = out_base + ".png"

    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  [OK] Saved  {png_path}")
    print(f"  [OK] Saved  {pdf_path}")


# ----------------------------------------------------------------
#  Main
# ----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate per-percentage tamper attack grids "
                    "(6 attacks x 5 stages) for journal review."
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-run attacks even if results already exist.")
    parser.add_argument("--image", type=str, default=None,
                        help="Process only this image basename (e.g. Lake).")
    parser.add_argument("--pct", type=int, default=None, nargs="+",
                        help="Process only these percentages (e.g. --pct 10 20).")
    args = parser.parse_args()

    setup_directories()

    # Discover source images
    source_files = []
    for ext in ["*.tiff", "*.png", "*.jpg", "*.jpeg"]:
        source_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

    if not source_files:
        print(f"ERROR: No source images found in '{INPUT_DIR}'. Aborting.")
        return

    percentages = args.pct if args.pct else PERCENTAGES

    for file_path in source_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        if args.image and base_name.lower() != args.image.lower():
            continue

        print(f"\n{'='*60}")
        print(f"  IMAGE: {base_name}")
        print(f"{'='*60}")

        for pct in percentages:
            print(f"\n-- {pct}% ----------------------------")

            ok = run_attacks_for_image_pct(file_path, base_name, pct,
                                           force=args.force)
            if not ok:
                print(f"  SKIP: pipeline failed for {base_name} @ {pct}%")
                continue

            print(f"  [grid] building {base_name} @ {pct}%")
            create_grid(base_name, pct)

    print("\n\nAll grids generated successfully!")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
