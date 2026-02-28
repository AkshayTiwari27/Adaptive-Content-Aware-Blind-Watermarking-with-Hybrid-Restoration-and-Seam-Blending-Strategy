import cv2
import os
import glob
import time
import numpy as np
import pandas as pd
import math
from skimage.metrics import structural_similarity as ssim
import my_custom_method as watermark_system

# --- Configuration ---
INPUT_DIR = "grayscale_normalized"
MAIN_RESULTS_DIR = "batch_results_png"
ATTACK_TYPES = ["content_removal", "copy_move", "splicing", "jpeg_compression", "noise", "cropping"]

# --- Metric Functions ---
def calculate_psnr(img1, img2):
    if img1 is None or img2 is None: return np.nan
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0: return 100.0
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_ssim(img1, img2):
    if img1 is None or img2 is None: return np.nan
    return ssim(img1, img2, data_range=255)

def calculate_ncc(img1, img2):
    if img1 is None or img2 is None: return np.nan
    i1, i2 = img1.astype(np.float64), img2.astype(np.float64)
    den = np.sqrt(np.sum(i1 ** 2)) * np.sqrt(np.sum(i2 ** 2))
    return np.sum(i1 * i2) / den if den != 0 else 0

def extract_watermark_layer(img):
    """Extracts the 2 LSBs used as the watermark payload in my_custom_method.py"""
    if img is None: return None
    return img & 0x03

def add_average_row(df, format_dict):
    """Calculates the column averages and appends the AVERAGE row."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    avg_vals = df[numeric_cols].mean()
    
    # Create a new row for the average
    avg_row = {col: avg_vals[col] for col in numeric_cols}
    avg_row['Image'] = 'AVERAGE'
    
    # Append the row
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    
    # Apply formatting
    for col, decimals in format_dict.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.{decimals}f}" if pd.notnull(x) else "N/A")
            
    return df

def get_images():
    files = []
    for ext in ['*.tiff', '*.png', '*.jpg', '*.jpeg']:
        files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    return sorted(files)

def main():
    print("--- Starting Quantitative Evaluation ---")
    
    files = get_images()
    if not files:
        print(f"No original images found in {INPUT_DIR}.")
        return

    # Data structures for our 4 tables
    table1_data = [] # PSNR & SSIM of Watermarked
    table2_data = [] # Watermark Robustness: NCC (Original WM vs Extracted WM)
    table3_data = [] # Recovery Quality: PSNR & SSIM
    table4_data = [] # Computational Time

    for file_path in files:
        original_filename = os.path.basename(file_path)
        base_name = os.path.splitext(original_filename)[0]
        png_filename = f"{base_name}.png"
        
        # Load Original and Watermarked
        orig_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        wm_path = os.path.join(MAIN_RESULTS_DIR, "0_Watermarked", png_filename)
        wm_img = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)

        if orig_img is None or wm_img is None:
            continue

        # Extract Original Watermark Payload (The 2 LSBs)
        orig_wm_layer = extract_watermark_layer(wm_img)

        # --- TABLE 1: Watermarked Images Quality ---
        psnr_wm = calculate_psnr(orig_img, wm_img)
        ssim_wm = calculate_ssim(orig_img, wm_img)
        table1_data.append({"Image": base_name, "PSNR (dB)": psnr_wm, "SSIM": ssim_wm})

        # Row dictionaries for Tables 2, 3, 4
        row_t2 = {"Image": base_name}
        row_t3 = {"Image": base_name}
        row_t4 = {"Image": base_name}

        for atk in ATTACK_TYPES:
            atk_path = os.path.join(MAIN_RESULTS_DIR, atk, "Attacked", png_filename)
            rec_path = os.path.join(MAIN_RESULTS_DIR, atk, "Recovered", png_filename)
            
            # --- Table 4: Computational Time (and cleanup) ---
            start_time = time.time()
            if os.path.exists(atk_path):
                watermark_system.recover(atk_path, "temp_rec.png") 
            end_time = time.time()
            row_t4[atk] = end_time - start_time
            
            # Clean up temporary files left behind
            if os.path.exists("temp_rec.png"): 
                os.remove("temp_rec.png")
            if os.path.exists("final_tamper_map.png"): 
                os.remove("final_tamper_map.png")

            # Load Attacked & Recovered Images
            atk_img = cv2.imread(atk_path, cv2.IMREAD_GRAYSCALE)
            rec_img = cv2.imread(rec_path, cv2.IMREAD_GRAYSCALE)
            
            # --- Table 2: Watermark Robustness (NCC of Watermarks) ---
            if atk_img is not None:
                ext_wm_layer = extract_watermark_layer(atk_img)
                row_t2[atk] = calculate_ncc(orig_wm_layer, ext_wm_layer)
            else:
                row_t2[atk] = np.nan

            # --- Table 3: Recovery Quality (PSNR, SSIM) ---
            if rec_img is not None:
                row_t3[f"{atk}_PSNR"] = calculate_psnr(orig_img, rec_img)
                row_t3[f"{atk}_SSIM"] = calculate_ssim(orig_img, rec_img)
            else:
                row_t3[f"{atk}_PSNR"] = np.nan
                row_t3[f"{atk}_SSIM"] = np.nan

        table2_data.append(row_t2)
        table3_data.append(row_t3)
        table4_data.append(row_t4)

    # Convert to DataFrames
    df1 = pd.DataFrame(table1_data)
    df2 = pd.DataFrame(table2_data)
    df3 = pd.DataFrame(table3_data)
    df4 = pd.DataFrame(table4_data)

    # Column ordering for Table 3 (grouped by attack)
    t3_cols = ["Image"]
    for atk in ATTACK_TYPES:
        t3_cols.extend([f"{atk}_PSNR", f"{atk}_SSIM"])
    df3 = df3[t3_cols]

    # --- Apply Formatting and Add AVERAGES ---
    df1 = add_average_row(df1, {"PSNR (dB)": 2, "SSIM": 4})
    df2 = add_average_row(df2, {atk: 4 for atk in ATTACK_TYPES})
    df3 = add_average_row(df3, {col: (2 if "PSNR" in col else 4) for col in t3_cols if col != "Image"})
    df4 = add_average_row(df4, {atk: 3 for atk in ATTACK_TYPES})

    # --- Print and Save ---
    print("\n" + "="*60)
    print("Table 1: PSNR and SSIM of Watermarked Images")
    print("="*60)
    print(df1.to_string(index=False))
    df1.to_csv(os.path.join(MAIN_RESULTS_DIR, "Table_1_Watermarked_Quality.csv"), index=False)

    print("\n" + "="*80)
    print("Table 2: Watermark Robustness: NCC (Original WM vs Extracted WM)")
    print("="*80)
    print(df2.to_string(index=False))
    df2.to_csv(os.path.join(MAIN_RESULTS_DIR, "Table_2_Watermark_Robustness_NCC.csv"), index=False)

    print("\n" + "="*120)
    print("Table 3: Quantitative Evaluation: Recovered PSNR (dB) and SSIM Under Different Attacks")
    print("="*120)
    print(df3.to_string(index=False))
    df3.to_csv(os.path.join(MAIN_RESULTS_DIR, "Table_3_Recovery_PSNR_SSIM.csv"), index=False)

    print("\n" + "="*80)
    print("Table 4: Computational Efficiency (Seconds)")
    print("="*80)
    print(df4.to_string(index=False))
    df4.to_csv(os.path.join(MAIN_RESULTS_DIR, "Table_4_Computational_Time.csv"), index=False)
    
    print(f"\nAll 4 tables have been saved as CSV files in the '{MAIN_RESULTS_DIR}' directory.")

if __name__ == "__main__":
    main()