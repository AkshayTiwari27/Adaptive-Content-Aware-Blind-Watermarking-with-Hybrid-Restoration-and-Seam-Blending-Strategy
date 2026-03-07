import cv2
import numpy as np
import os
import glob
import math
import csv
from skimage.metrics import structural_similarity as compare_ssim

# Import your existing modules
import my_custom_method as watermark_system
import attack_image as attacker

INPUT_DIR = "grayscale_normalized"
RESULTS_DIR = "robustness_results"

# --- Define the specific testing parameters from the tables ---
JPEG_QUALITIES = [90, 70, 50, 30, 10]
NOISE_DENSITIES = [0.01, 0.03, 0.05, 0.07, 0.09]

# --- Output CSV filenames ---
JPEG_CSV = "Table_7_JPEG_Compression.csv"
NOISE_CSV = "Table_9_Salt_Pepper_Noise.csv"

def setup_directories():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "0_Watermarked"), exist_ok=True)
    
    for q in JPEG_QUALITIES:
        os.makedirs(os.path.join(RESULTS_DIR, f"JPEG_Q{q}", "Attacked"), exist_ok=True)
        os.makedirs(os.path.join(RESULTS_DIR, f"JPEG_Q{q}", "Recovered"), exist_ok=True)
        
    for d in NOISE_DENSITIES:
        pct = int(d * 100)
        os.makedirs(os.path.join(RESULTS_DIR, f"Noise_{pct}pct", "Attacked"), exist_ok=True)
        os.makedirs(os.path.join(RESULTS_DIR, f"Noise_{pct}pct", "Recovered"), exist_ok=True)

def calculate_psnr(img1, img2):
    if img1 is None or img2 is None: return 0.0
    if img1.shape != img2.shape: img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # FIXED: Cast to float64 to prevent uint8 integer overflow/underflow
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    
    if mse == 0: return 100.0
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_ssim(img1, img2):
    if img1 is None or img2 is None: return 0.0
    if img1.shape != img2.shape: img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Grayscale SSIM evaluation
    if len(img1.shape) == 3:
        score, _ = compare_ssim(img1, img2, full=True, channel_axis=-1)
    else:
        score, _ = compare_ssim(img1, img2, full=True, data_range=255) # Added data_range=255
    return score

def run_robustness_test():
    setup_directories()
    
    files = []
    for ext in ['*.tiff', '*.png', '*.jpg', '*.jpeg']:
        files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
        
    # Data structures to hold results for the tables
    jpeg_results = {}
    noise_results = {}
    
    print("--- Starting Robustness Evaluation ---")

    for file_path in files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        png_filename = f"{base_name}.png"
        print(f"\nProcessing Image: {png_filename}")
        
        jpeg_results[base_name] = {}
        noise_results[base_name] = {}
        
        # FIXED: Evaluate against original grayscale
        original_img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        # Embedder usually requires the original path
        wm_save_path = os.path.join(RESULTS_DIR, "0_Watermarked", png_filename)
        
        if not watermark_system.embed(file_path, wm_save_path):
            continue
            
        # We need the color image for the attackers to process correctly
        wm_img_color = cv2.imread(wm_save_path)

        # ---------------------------------------------------------
        # 1. TABLE 7: JPEG Compression (Q=90 to Q=10)
        # ---------------------------------------------------------
        print("  -> Testing JPEG Compression (Table 7):")
        for q in JPEG_QUALITIES:
            print(f"     -> Q={q} ...", end=" ")
            
            # FIXED: Properly nest attack, save, recover, evaluate inside the loop
            attacked_img, _ = attacker.attack_jpeg_compression(wm_img_color, quality=q)
            atk_dir = os.path.join(RESULTS_DIR, f"JPEG_Q{q}")
            
            atk_save_path = os.path.join(atk_dir, "Attacked", png_filename)
            cv2.imwrite(atk_save_path, attacked_img)

            rec_save_path = os.path.join(atk_dir, "Recovered", png_filename)
            watermark_system.recover(atk_save_path, rec_save_path)

            # FIXED: Load recovered image in grayscale to evaluate correctly
            rec_img_gray = cv2.imread(rec_save_path, cv2.IMREAD_GRAYSCALE)
            
            psnr = calculate_psnr(original_img_gray, rec_img_gray)
            jpeg_results[base_name][q] = psnr
            print(f"PSNR: {psnr:.2f} dB")

        # ---------------------------------------------------------
        # 2. TABLE 9: Salt & Pepper Noise (0.01 to 0.09)
        # ---------------------------------------------------------
        print("  -> Testing Salt & Pepper Noise (Table 9):")
        for d in NOISE_DENSITIES:
            pct = int(d * 100)
            print(f"     -> Density={d} ...", end=" ")
            
            # FIXED: Properly nest attack, save, recover, evaluate inside the loop
            attacked_img, _ = attacker.attack_salt_and_pepper(wm_img_color, amount=d)
            atk_dir = os.path.join(RESULTS_DIR, f"Noise_{pct}pct")
            
            atk_save_path = os.path.join(atk_dir, "Attacked", png_filename)
            cv2.imwrite(atk_save_path, attacked_img)

            rec_save_path = os.path.join(atk_dir, "Recovered", png_filename)
            watermark_system.recover(atk_save_path, rec_save_path)

            # FIXED: Load recovered image in grayscale to evaluate correctly
            rec_img_gray = cv2.imread(rec_save_path, cv2.IMREAD_GRAYSCALE)
            
            psnr = calculate_psnr(original_img_gray, rec_img_gray)
            ssim_val = calculate_ssim(original_img_gray, rec_img_gray)
            
            noise_results[base_name][d] = (psnr, ssim_val)
            print(f"PSNR: {psnr:.2f} dB | SSIM: {ssim_val:.4f}")

    # ==========================================
    # CSV EXPORT ROUTINES
    # ==========================================
    
    # Export Table 7
    print(f"\n--- Exporting {JPEG_CSV} ---")
    with open(JPEG_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        headers = ["Image"] + [f"Q={q}" for q in JPEG_QUALITIES]
        writer.writerow(headers)
        
        avg_psnrs = {q: [] for q in JPEG_QUALITIES}
        
        for img_name, data in jpeg_results.items():
            row = [img_name]
            for q in JPEG_QUALITIES:
                val = data.get(q, 0.0)
                row.append(f"{val:.2f}")
                avg_psnrs[q].append(val)
            writer.writerow(row)
            
        # Write Average Row
        writer.writerow([])
        avg_row = ["AVERAGE"]
        for q in JPEG_QUALITIES:
            avg_val = sum(avg_psnrs[q]) / len(avg_psnrs[q]) if avg_psnrs[q] else 0.0
            avg_row.append(f"{avg_val:.2f}")
        writer.writerow(avg_row)

    # Export Table 9
    print(f"--- Exporting {NOISE_CSV} ---")
    with open(NOISE_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        headers = ["Image"] + [str(d) for d in NOISE_DENSITIES]
        writer.writerow(headers)
        
        avg_noise = {d: {"psnr": [], "ssim": []} for d in NOISE_DENSITIES}
        
        for img_name, data in noise_results.items():
            row = [img_name]
            for d in NOISE_DENSITIES:
                psnr, ssim_val = data.get(d, (0.0, 0.0))
                row.append(f"{psnr:.2f} / {ssim_val:.4f}")
                avg_noise[d]["psnr"].append(psnr)
                avg_noise[d]["ssim"].append(ssim_val)
            writer.writerow(row)

        # Write Average Row
        writer.writerow([])
        avg_row = ["AVERAGE"]
        for d in NOISE_DENSITIES:
            avg_p = sum(avg_noise[d]["psnr"]) / len(avg_noise[d]["psnr"]) if avg_noise[d]["psnr"] else 0.0
            avg_s = sum(avg_noise[d]["ssim"]) / len(avg_noise[d]["ssim"]) if avg_noise[d]["ssim"] else 0.0
            avg_row.append(f"{avg_p:.2f} / {avg_s:.4f}")
        writer.writerow(avg_row)

    print("\nProcessing Complete! Check the CSV files and the 'robustness_results' directory.")

if __name__ == "__main__":
    run_robustness_test()