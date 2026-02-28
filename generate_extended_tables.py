import cv2
import os
import glob
import numpy as np
import pandas as pd
import math
import atexit
from skimage.metrics import structural_similarity as ssim

# Import your custom modules
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

def calculate_psnr(img1, img2):
    img1, img2 = _ensure_gray(img1), _ensure_gray(img2)
    if img1 is None or img2 is None: return np.nan
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0: return 100.0
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_ssim(img1, img2):
    img1, img2 = _ensure_gray(img1), _ensure_gray(img2)
    if img1 is None or img2 is None: return np.nan
    return ssim(img1, img2, data_range=255)

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
    avg_row['Image'] = 'AVERAGE'
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
    print("--- Starting Extended Quantitative Evaluation ---")
    files = get_images()
    if not files:
        print(f"No original images found in {INPUT_DIR}.")
        return

    table5_data = [] 
    table6_data = [] 
    table7_data = [] 
    
    tdr_accumulators = {
        "Copy-Move": [], "Splicing": [], "Content Removal": [],
        "Cropping (40%)": [], "JPEG (Q=50)": [], "Salt & Pepper (0.05)": []
    }

    for file_path in files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        orig_img = cv2.imread(file_path)
        wm_path = os.path.join(WM_DIR, f"{base_name}.png")
        wm_img = cv2.imread(wm_path)

        if orig_img is None or wm_img is None:
            continue
            
        print(f"Processing image: {base_name}...")
        orig_wm_layer = extract_watermark_layer(wm_img)

        # TABLE 5: Varying Tampering Rates
        row_t5 = {"Image": base_name}
        for rate in [0, 10, 20, 30, 40, 50, 60, 70]:
            if rate == 0:
                row_t5["0%"] = f"{calculate_psnr(orig_img, wm_img):.2f} / {calculate_ssim(orig_img, wm_img):.4f}"
            else:
                atk_img, _ = attacker.attack_cropping(wm_img, percent=rate)
                cv2.imwrite(temp_atk_path, atk_img)
                watermark_system.recover(temp_atk_path, temp_rec_path)
                rec_img = cv2.imread(temp_rec_path)
                
                psnr_val = calculate_psnr(orig_img, rec_img)
                ssim_val = calculate_ssim(orig_img, rec_img)
                row_t5[f"{rate}%"] = f"{psnr_val:.2f} / {ssim_val:.4f}"
        table5_data.append(row_t5)

        # TABLE 6: NCC S&P Noise
        row_t6 = {"Image": base_name}
        for density in [0.01, 0.03, 0.05, 0.10]:
            atk_img, _ = attacker.attack_salt_and_pepper(wm_img, amount=density)
            ext_wm_layer = extract_watermark_layer(atk_img)
            row_t6[f"{density}"] = calculate_ncc(orig_wm_layer, ext_wm_layer)
        table6_data.append(row_t6)

        # TABLE 7: PSNR JPEG Compression
        row_t7 = {"Image": base_name}
        for q in [90, 70, 50, 30, 10]:
            atk_img, _ = attacker.attack_jpeg_compression(wm_img, quality=q)
            cv2.imwrite(temp_atk_path, atk_img)
            watermark_system.recover(temp_atk_path, temp_rec_path)
            rec_img = cv2.imread(temp_rec_path)
            row_t7[f"Q={q}"] = calculate_psnr(orig_img, rec_img)
        table7_data.append(row_t7)

        # TABLE 8: TDR Accumulation
        attacks_for_tdr = {
            "Copy-Move": attacker.attack_copy_move(wm_img)[0],
            "Splicing": attacker.attack_political_splicing(wm_img)[0],
            "Content Removal": attacker.attack_content_removal(wm_img)[0],
            "Cropping (40%)": attacker.attack_cropping(wm_img, percent=40)[0],
            "JPEG (Q=50)": attacker.attack_jpeg_compression(wm_img, quality=50)[0],
            "Salt & Pepper (0.05)": attacker.attack_salt_and_pepper(wm_img, amount=0.05)[0]
        }

        for atk_name, atk_img in attacks_for_tdr.items():
            if atk_img is not None:
                cv2.imwrite(temp_atk_path, atk_img)
                watermark_system.recover(temp_atk_path, temp_rec_path)
                tdr_val = calculate_tdr(wm_img, atk_img, tamper_map_path)
                tdr_accumulators[atk_name].append(tdr_val)

    # --- Construct DataFrames & Calculate Averages ---
    
    # Table 5 (Custom string splitting logic for average)
    df5 = pd.DataFrame(table5_data)
    avg_row_t5 = {"Image": "AVERAGE"}
    rate_cols = ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%"]
    for col in rate_cols:
        p_sum, s_sum, count = 0.0, 0.0, 0
        if col in df5.columns:
            for val in df5[col]:
                if pd.notnull(val) and " / " in str(val):
                    p_str, s_str = str(val).split(" / ")
                    if p_str != "nan" and s_str != "nan":
                        p_sum += float(p_str)
                        s_sum += float(s_str)
                        count += 1
            avg_row_t5[col] = f"{p_sum/count:.2f} / {s_sum/count:.4f}" if count > 0 else "N/A"
    df5 = pd.concat([df5, pd.DataFrame([avg_row_t5])], ignore_index=True)
    
    # Table 6 & 7 (Standard numeric averages)
    df6 = pd.DataFrame(table6_data)
    df6 = add_average_row(df6, {col: lambda x: f"{x:.4f}" for col in ["0.01", "0.03", "0.05", "0.1"]})

    df7 = pd.DataFrame(table7_data)
    df7 = add_average_row(df7, {col: lambda x: f"{x:.2f}" for col in ["Q=90", "Q=70", "Q=50", "Q=30", "Q=10"]})

    # Table 8 (Average TDR calculation)
    table8_data = []
    all_tdrs = []
    for atk_name, tdr_list in tdr_accumulators.items():
        valid_tdrs = [x for x in tdr_list if not np.isnan(x)]
        avg_tdr = np.mean(valid_tdrs) if valid_tdrs else np.nan
        table8_data.append({"Attack Type": atk_name, "TDR (%)": avg_tdr})
        all_tdrs.extend(valid_tdrs)
        
    df8 = pd.DataFrame(table8_data)
    overall_avg_tdr = np.mean(all_tdrs) if all_tdrs else np.nan
    df8 = pd.concat([df8, pd.DataFrame([{"Attack Type": "AVERAGE", "TDR (%)": overall_avg_tdr}])], ignore_index=True)
    df8["TDR (%)"] = df8["TDR (%)"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")

    # --- Print and Save ---
    print("\n" + "="*100)
    print("Table 5: Detailed Recovery Performance (PSNR / SSIM) Under Varying Tampering Rates")
    print("="*100)
    print(df5.to_string(index=False))
    df5.to_csv(os.path.join(MAIN_RESULTS_DIR, "Table_5_Recovery_Varying_Rates.csv"), index=False)

    print("\n" + "="*60)
    print("Table 6: NCC Under Salt & Pepper Noise")
    print("="*60)
    print(df6.to_string(index=False))
    df6.to_csv(os.path.join(MAIN_RESULTS_DIR, "Table_6_NCC_Noise.csv"), index=False)

    print("\n" + "="*70)
    print("Table 7: Robustness Under JPEG Compression (PSNR dB)")
    print("="*70)
    print(df7.to_string(index=False))
    df7.to_csv(os.path.join(MAIN_RESULTS_DIR, "Table_7_JPEG_Compression.csv"), index=False)

    print("\n" + "="*50)
    print("Table 8: Tamper Detection Rate (TDR)")
    print("="*50)
    print(df8.to_string(index=False))
    df8.to_csv(os.path.join(MAIN_RESULTS_DIR, "Table_8_Tamper_Detection_Rate.csv"), index=False)

    print(f"\nAll Extended Tables have been saved in '{MAIN_RESULTS_DIR}'.")

if __name__ == "__main__":
    main()