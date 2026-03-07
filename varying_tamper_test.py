import cv2
import numpy as np
import os
import glob
import math
import csv

# Import your existing modules
import my_custom_method as watermark_system
import attack_image as attacker

INPUT_DIR = "grayscale_normalized"
RESULTS_DIR = "varying_tamper_results"
OUTPUT_CSV = "Table_5_Recovery_Varying_Rates_Generated.csv"

ATTACKS = ["content_removal", "copy_move", "splicing", "cropping"]
PERCENTAGES = [10, 20, 30, 40, 50]

def setup_directories():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "0_Watermarked"), exist_ok=True)
    for atk in ATTACKS:
        for p in PERCENTAGES:
            os.makedirs(os.path.join(RESULTS_DIR, f"{atk}_{p}pct", "Recovered"), exist_ok=True)
            os.makedirs(os.path.join(RESULTS_DIR, f"{atk}_{p}pct", "Attacked"), exist_ok=True)

def calculate_psnr(img1, img2):
    if img1 is None or img2 is None: return 0
    if img1.shape != img2.shape: img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * math.log10(255.0 / math.sqrt(mse))

def get_attack_dimensions(rows, cols, percent):
    """Calculates width and height proportionally to cover exactly 'percent' of the area."""
    ratio = math.sqrt(percent / 100.0)
    side_w = int(cols * ratio)
    side_h = int(rows * ratio)
    return side_w, side_h

def apply_percent_attack(attack_name, image, percent):
    rows, cols = image.shape[:2]
    
    if attack_name == "cropping":
        return attacker.attack_cropping(image, percent=percent)
        
    w, h = get_attack_dimensions(rows, cols, percent)
    
    if attack_name == "content_removal":
        x = (cols - w) // 2
        y = (rows - h) // 2
        return attacker.attack_content_removal(image, x=x, y=y, w=w, h=h)
        
    elif attack_name == "copy_move":
        # Explicitly separate source and destination to prevent overlap
        # Source: Top-Left, Destination: Bottom-Right
        src_x, src_y = 0, 0
        dst_x = cols - w
        dst_y = rows - h
        return attacker.attack_copy_move(image, src_x=src_x, src_y=src_y, w=w, h=h, dst_x=dst_x, dst_y=dst_y)
        
    elif attack_name == "splicing":
        x = (cols - w) // 2
        y = (rows - h) // 2
        return attacker.attack_political_splicing(image, x=x, y=y, w=w, h=h)

    return None, None

def run_varying_tamper_test():
    setup_directories()
    
    files = []
    for ext in ['*.tiff', '*.png', '*.jpg', '*.jpeg']:
        files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
        
    results = {}
    
    print("--- Starting Varying Tampering Rate Evaluation ---")

    for file_path in files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        png_filename = f"{base_name}.png"
        print(f"\nProcessing Image: {png_filename}")
        
        results[png_filename] = {atk: {} for atk in ATTACKS}
        original_img = cv2.imread(file_path)
        
        wm_save_path = os.path.join(RESULTS_DIR, "0_Watermarked", png_filename)
        if not watermark_system.embed(file_path, wm_save_path):
            continue
            
        wm_img = cv2.imread(wm_save_path)

        for atk in ATTACKS:
            print(f"  -> Attack: {atk}")
            for pct in PERCENTAGES:
                print(f"     -> {pct}% ...", end=" ")
                
                attacked_img, _ = apply_percent_attack(atk, wm_img, pct)
                if attacked_img is None:
                    results[png_filename][atk][pct] = 0.0
                    print("Failed.")
                    continue

                atk_save_path = os.path.join(RESULTS_DIR, f"{atk}_{pct}pct", "Attacked", png_filename)
                cv2.imwrite(atk_save_path, attacked_img)

                rec_save_path = os.path.join(RESULTS_DIR, f"{atk}_{pct}pct", "Recovered", png_filename)
                watermark_system.recover(atk_save_path, rec_save_path)

                rec_img = cv2.imread(rec_save_path)
                psnr = calculate_psnr(original_img, rec_img)
                results[png_filename][atk][pct] = psnr
                
                print(f"PSNR: {psnr:.2f} dB")

    print(f"\n--- Exporting Results to {OUTPUT_CSV} ---")
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        headers = ["Image", "Attack"] + [f"{p}%_PSNR" for p in PERCENTAGES]
        writer.writerow(headers)
        
        percentage_totals = {p: [] for p in PERCENTAGES}
        
        for img_name, atk_data in results.items():
            img_display = img_name.split('.')[0]
            for i, atk in enumerate(ATTACKS):
                row = [img_display if i == 0 else ""]
                row.append(atk.replace("_", " ").title())
                
                for pct in PERCENTAGES:
                    psnr_val = atk_data[atk].get(pct, 0.0)
                    row.append(f"{psnr_val:.2f}")
                    if psnr_val > 0:
                        percentage_totals[pct].append(psnr_val)
                
                writer.writerow(row)
        
        writer.writerow([]) 
        avg_row = ["AVERAGE", "-"]
        for pct in PERCENTAGES:
            vals = percentage_totals[pct]
            avg_val = sum(vals) / len(vals) if vals else 0.0
            avg_row.append(f"{avg_val:.2f}")
            
        writer.writerow(avg_row)

    print("Testing Complete!")

if __name__ == "__main__":
    run_varying_tamper_test()