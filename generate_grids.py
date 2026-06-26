"""
generate_grids.py
==============================================
Reads from dataset_results/ and generates one publication-quality grid
per image:
    5 rows (attacks) x 5 columns (stages)

Grid style matches generate_varying_tamper_grids.py exactly:
    figsize=(15, 16), dpi=300, wspace=0.05, hspace=0.05
    grayscale display, black cell borders, rotation=0 row labels

Columns: (a) Original | (b) Watermarked | (c) Attacked | (d) Tamper Map | (e) Recovered
Rows   : Content Removal | Copy-Move | Splicing | JPEG (Q=70) | Salt & Pepper (5%)

Output:
    grids/
        {Dataset}/
            Grid_{Dataset}_{image_name}.png
            Grid_{Dataset}_{image_name}.pdf

Usage:
    python generate_grids.py
"""

import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Force UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = "dataset_results"
GRIDS_DIR   = "grids"

ATTACK_KEYS = ["content_removal", "copy_move", "splicing", "jpeg_q70", "snp_005"]

ROW_LABELS = {
    "content_removal": "Content\nRemoval",
    "copy_move":       "Copy-Move",
    "splicing":        "Splicing",
    "jpeg_q70":        "JPEG\n(Q=70)",
    "snp_005":         "Salt &\nPepper (5%)",
}

COL_LABELS = [
    "(a) Original",
    "(b) Watermarked",
    "(c) Attacked",
    "(d) Tamper Map",
    "(e) Recovered",
]


# ---------------------------------------------------------------------------
# Grid builder — matches generate_varying_tamper_grids.py style exactly
# ---------------------------------------------------------------------------
def create_grid(ds_name, img_name, img_dir, out_dir):
    # Load original and watermarked in grayscale for display
    orig_img = cv2.imread(os.path.join(img_dir, "orig.png"), cv2.IMREAD_GRAYSCALE)
    wmk_img  = cv2.imread(os.path.join(img_dir, "wmk.png"),  cv2.IMREAD_GRAYSCALE)

    n_rows = len(ATTACK_KEYS)   # 5
    n_cols = 5

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=(15, 16), dpi=300)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for row_idx, atk_key in enumerate(ATTACK_KEYS):
        atk_img = cv2.imread(
            os.path.join(img_dir, f"{atk_key}_atk.png"), cv2.IMREAD_GRAYSCALE
        )
        map_img = cv2.imread(
            os.path.join(img_dir, f"{atk_key}_map.png"), cv2.IMREAD_GRAYSCALE
        )
        rec_img = cv2.imread(
            os.path.join(img_dir, f"{atk_key}_rec.png"), cv2.IMREAD_GRAYSCALE
        )

        cell_images = [orig_img, wmk_img, atk_img, map_img, rec_img]

        for col_idx, img in enumerate(cell_images):
            ax = axes[row_idx, col_idx]

            if img is not None:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            else:
                ax.text(
                    0.5, 0.5, "Missing",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=10, color="red",
                )

            ax.set_xticks([])
            ax.set_yticks([])

            # Black border around every cell (matching Grid_Lake style)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
                spine.set_edgecolor("black")

            # Column header — top row only
            if row_idx == 0:
                ax.set_title(COL_LABELS[col_idx], fontsize=14, pad=10)

            # Row label — first column only
            if col_idx == 0:
                ax.set_ylabel(
                    ROW_LABELS[atk_key],
                    fontsize=12, labelpad=25,
                    rotation=0, ha="right", va="center",
                )

    os.makedirs(out_dir, exist_ok=True)
    base_out = os.path.join(out_dir, f"Grid_{ds_name}_{img_name}")
    plt.savefig(base_out + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(base_out + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {base_out}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(RESULTS_DIR):
        print(
            f"ERROR: '{RESULTS_DIR}/' not found. "
            "Run experiment_all_datasets.py first."
        )
        return

    for ds_name in sorted(os.listdir(RESULTS_DIR)):
        ds_path = os.path.join(RESULTS_DIR, ds_name)
        # Skip files (e.g. summary.csv) and the cache dir
        if not os.path.isdir(ds_path):
            continue

        print(f"\n[{ds_name}]")
        out_dir = os.path.join(GRIDS_DIR, ds_name)

        for img_name in sorted(os.listdir(ds_path)):
            img_dir = os.path.join(ds_path, img_name)
            if not os.path.isdir(img_dir):
                continue
            create_grid(ds_name, img_name, img_dir, out_dir)

    print(f"\nAll grids done. Output in: {os.path.abspath(GRIDS_DIR)}/")


if __name__ == "__main__":
    main()
