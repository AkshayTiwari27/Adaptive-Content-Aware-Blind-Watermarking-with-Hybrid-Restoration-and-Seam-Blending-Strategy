import cv2
import os
import glob
import numpy as np
import pandas as pd
import math
import atexit

# Import custom modules
import my_custom_method as watermark_system
import attack_image as attacker

# --- Configuration ---
INPUT_DIR = "grayscale_normalized"
MAIN_RESULTS_DIR = "batch_results_png"
WM_DIR = os.path.join(MAIN_RESULTS_DIR, "0_Watermarked")

temp_atk_path = "temp_extended_atk.png"
temp_rec_path = "temp_extended_rec.png"
tamper_map_path = "final_tamper_map.png"

# --- Guaranteed Cleanup ---
def cleanup_temp_files():
    """Ensures temp files are deleted even if the script crashes mid-execution."""
    for temp_file in [temp_atk_path, temp_rec_path, tamper_map_path]:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass

atexit.register(cleanup_temp_files)

# --- Metric Helper Functions ---
def _ensure_gray(img):
    """Safely converts 3-channel images to 1-channel grayscale for accurate math."""
    if img is None: return None
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def calculate_ncc(img1, img2):
    img1, img2 = _ensure_gray(img1), _ensure_gray(img2)
    if img1 is None or img2 is None: return np.nan
    i1, i2 = img1.astype(np.float64), img2.astype(np.float64)
    den = np.sqrt(np.sum(i1 ** 2)) * np.sqrt(np.sum(i2 ** 2))
    return np.sum(i1 * i2) / den if den != 0 else 0

def extract_watermark_layer(img):
    img = _ensure_gray(img)
    if img is None: return None
    return img & 0x03

def calculate_tdr(wm_img, atk_img, det_map_path):
    """Calculates Tamper Detection Rate (TDR) in %"""
    wm_img, atk_img = _ensure_gray(wm_img), _ensure_gray(atk_img)
    if wm_img is None or atk_img is None or not os.path.exists(det_map_path):
        return np.nan
        
    diff = cv2.absdiff(wm_img, atk_img)
    gt_map = np.where(diff > 0, 255, 0).astype(np.uint8)
    
    det_map = cv2.imread(det_map_path, cv2.IMREAD_GRAYSCALE)
    if det_map is None: return np.nan
    det_map = np.where(det_map > 127, 255, 0).astype(np.uint8)
    
    actual_tampered = np.sum(gt_map == 255)
    if actual_tampered == 0:
        return 100.0 
        
    true_positives = np.sum((gt_map == 255) & (det_map == 255))
    return (true_positives / actual_tampered) * 100.0

def add_average_row(df, format_dict=None):
    """Calculates standard numeric averages for DataFrames."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    avg_vals = df[numeric_cols].mean()
    avg_row = {col: avg_vals[col] for col in numeric_cols}
    
    # Ensure the first column (Image or Attack Type) gets the 'AVERAGE' label
    first_col = df.columns[0]
    avg_row[first_col] = 'AVERAGE'
    
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    
    if format_dict:
        for col, formatter in format_dict.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda x: formatter(x) if pd.notnull(x) else "N/A")
    return df

def get_images():
    files = []
    for ext in ['*.tiff', '*.png', '*.jpg', '*.jpeg']:
        files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    return sorted(files)

def main():
    print("--- Starting Targeted Quantitative Evaluation ---")
    files = get_images()
    if not files:
        print(f"No original images found in {INPUT_DIR}.")
        return

    table8_data = [] 
    
    tdr_accumulators = {
        "Content Removal": [],
        "Copy-Move": [],
        "Splicing": [],
        "Cropping (40%)": [],
        "JPEG Compression (Q=50)": [],
        "Salt & Pepper Noise (0.05)": []
    }

    sp_densities = [0.01, 0.03, 0.05, 0.10]

    for file_path in files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        wm_path = os.path.join(WM_DIR, f"{base_name}.png")
        wm_img = cv2.imread(wm_path)

        if wm_img is None:
            continue
            
        print(f"Processing image: {base_name}...")
        orig_wm_layer = extract_watermark_layer(wm_img)

        # --- TABLE 8: NCC Under Salt & Pepper Noise ---
        row_t8 = {"Image": base_name}
        for density in sp_densities:
            atk_img, _ = attacker.attack_salt_and_pepper(wm_img, amount=density)
            ext_wm_layer = extract_watermark_layer(atk_img)
            row_t8[f"{density}"] = calculate_ncc(orig_wm_layer, ext_wm_layer)
        table8_data.append(row_t8)

        # --- TABLE 10: TDR Accumulation ---
        attacks_for_tdr = {
            "Content Removal": attacker.attack_content_removal(wm_img)[0],
            "Copy-Move": attacker.attack_copy_move(wm_img)[0],
            "Splicing": attacker.attack_political_splicing(wm_img)[0],
            "Cropping (40%)": attacker.attack_cropping(wm_img, percent=40)[0],
            "JPEG Compression (Q=50)": attacker.attack_jpeg_compression(wm_img, quality=50)[0],
            "Salt & Pepper Noise (0.05)": attacker.attack_salt_and_pepper(wm_img, amount=0.05)[0]
        }

        for atk_name, atk_img in attacks_for_tdr.items():
            if atk_img is not None:
                cv2.imwrite(temp_atk_path, atk_img)
                watermark_system.recover(temp_atk_path, temp_rec_path)
                tdr_val = calculate_tdr(wm_img, atk_img, tamper_map_path)
                tdr_accumulators[atk_name].append(tdr_val)

    # --- Construct DataFrames & Calculate Averages ---
    
    # Table 8 Formatting
    df8 = pd.DataFrame(table8_data)
    df8 = add_average_row(df8, {col: lambda x: f"{x:.4f}" for col in ["0.01", "0.03", "0.05", "0.1"]})

    # Table 10 Formatting
    table10_data = []
    all_tdrs = []
    
    # Ensure the order matches exactly what you requested
    ordered_attacks = ["Content Removal", "Copy-Move", "Splicing", "Cropping (40%)", "JPEG Compression (Q=50)", "Salt & Pepper Noise (0.05)"]
    
    for atk_name in ordered_attacks:
        tdr_list = tdr_accumulators[atk_name]
        valid_tdrs = [x for x in tdr_list if not np.isnan(x)]
        avg_tdr = np.mean(valid_tdrs) if valid_tdrs else np.nan
        table10_data.append({"Attack Type": atk_name, "Average TDR (%)": avg_tdr})
        all_tdrs.extend(valid_tdrs)
        
    df10 = pd.DataFrame(table10_data)
    overall_avg_tdr = np.mean(all_tdrs) if all_tdrs else np.nan
    df10 = pd.concat([df10, pd.DataFrame([{"Attack Type": "AVERAGE", "Average TDR (%)": overall_avg_tdr}])], ignore_index=True)
    df10["Average TDR (%)"] = df10["Average TDR (%)"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")

    # --- Print and Save ---
    print("\n" + "="*70)
    print("Table 8: NCC Under Salt & Pepper Noise (0.01 to 0.10 Density)")
    print("="*70)
    print(df8.to_string(index=False))
    df8.to_csv(os.path.join(MAIN_RESULTS_DIR, "Table_8_NCC_Noise.csv"), index=False)

    print("\n" + "="*60)
    print("Table 10: Tamper Detection Rate (TDR) Under Different Attacks")
    print("="*60)
    print(df10.to_string(index=False))
    df10.to_csv(os.path.join(MAIN_RESULTS_DIR, "Table_10_Tamper_Detection_Rate.csv"), index=False)

    print(f"\nTables 8 and 10 have been saved in '{MAIN_RESULTS_DIR}'.")

if __name__ == "__main__":
    main()