import cv2
import matplotlib.pyplot as plt
import os
import glob

# Configuration matching your repository structure
INPUT_DIR = "grayscale_normalized"
MAIN_RESULTS_DIR = "batch_results_png"
OUTPUT_DIR = "visual_grids"

# The exact order from your PDF
ATTACK_TYPES = [
    "content_removal", 
    "copy_move", 
    "splicing", 
    "jpeg_compression", 
    "noise", 
    "cropping"
]

ATTACK_LABELS = [
    "Content Removal", 
    "Copy Move", 
    "Splicing", 
    "JPEG Compression", 
    "Noise", 
    "Cropping"
]

COL_LABELS = [
    "(a) Original", 
    "(b) Watermarked", 
    "(c) Attacked", 
    "(d) Tamper Map", 
    "(e) Recovered"
]

def create_grid_for_image(base_name, original_ext):
    """Generates a high-res grid for a single image across all attacks."""
    print(f"Generating grid for {base_name}...")
    
    # Create a figure with 6 rows (attacks) and 5 columns (stages)
    fig, axes = plt.subplots(nrows=len(ATTACK_TYPES), ncols=5, figsize=(15, 18))
    
    # Reduce whitespace between images
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # File paths for Original and Watermarked (constant across rows)
    orig_path = os.path.join(INPUT_DIR, f"{base_name}{original_ext}")
    wm_path = os.path.join(MAIN_RESULTS_DIR, "0_Watermarked", f"{base_name}.png")

    orig_img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
    wm_img = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)

    for row_idx, attack in enumerate(ATTACK_TYPES):
        # File paths for specific attacks
        atk_path = os.path.join(MAIN_RESULTS_DIR, attack, "Attacked", f"{base_name}.png")
        map_path = os.path.join(MAIN_RESULTS_DIR, attack, "Tamper_Maps", f"{base_name}.png")
        rec_path = os.path.join(MAIN_RESULTS_DIR, attack, "Recovered", f"{base_name}.png")

        # Load images
        atk_img = cv2.imread(atk_path, cv2.IMREAD_GRAYSCALE)
        map_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        rec_img = cv2.imread(rec_path, cv2.IMREAD_GRAYSCALE)

        # Assemble row
        images = [orig_img, wm_img, atk_img, map_img, rec_img]

        for col_idx, img in enumerate(images):
            ax = axes[row_idx, col_idx]
            
            if img is not None:
                # Display image; use cmap='gray' since the images are grayscale
                ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            else:
                # Blank space if file is missing
                ax.text(0.5, 0.5, 'Missing\nImage', ha='center', va='center')
                
            # Remove axis ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Add column labels to the top row 
            if row_idx == 0:
                ax.set_title(COL_LABELS[col_idx], fontsize=14, pad=10)

            # Add row labels to the first column [cite: 1, 2, 3, 4, 5, 6]
            if col_idx == 0:
                ax.set_ylabel(ATTACK_LABELS[row_idx], fontsize=12, labelpad=25, rotation=0, ha='right', va='center')

    # Save as high-resolution PDF and PNG
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_out = os.path.join(OUTPUT_DIR, f"Grid_{base_name}.pdf")
    png_out = os.path.join(OUTPUT_DIR, f"Grid_{base_name}.png")
    
    # bbox_inches='tight' removes excess white border
    plt.savefig(pdf_out, dpi=300, bbox_inches='tight')
    plt.savefig(png_out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved to {pdf_out}")

def main():
    # Find all original files in the input directory
    files = glob.glob(os.path.join(INPUT_DIR, '*.*'))
    
    if not files:
        print(f"No original images found in {INPUT_DIR}.")
        return

    for file_path in files:
        filename = os.path.basename(file_path)
        base_name, ext = os.path.splitext(filename)
        
        # Skip hidden files or non-images
        if ext.lower() not in ['.tiff', '.png', '.jpg', '.jpeg']:
            continue
            
        create_grid_for_image(base_name, ext)

if __name__ == "__main__":
    main()