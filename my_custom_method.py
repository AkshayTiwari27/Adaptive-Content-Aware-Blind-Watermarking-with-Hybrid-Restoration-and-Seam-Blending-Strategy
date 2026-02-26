import cv2
import numpy as np
import hashlib
import random
import os

print("--- Smart Self-Recovery System (DLSBM v13 - High Clarity) ---")
print("--- [Fixed: Smart Median for Sharp Noise Repair + Tight Seam Blending] ---")

# --- CONFIGURATION ---
BLOCK_SIZE = 4   # 4x4 Blocks
KEY = 9999       # Secret key

def get_random_mapping(total_blocks, key):
    np.random.seed(key)
    indices = np.arange(total_blocks)
    np.random.shuffle(indices)
    return indices

def get_location_dependent_hash(flat_block, block_index):
    data = flat_block.tobytes()
    index_bytes = int(block_index).to_bytes(4, byteorder='big')
    full_hash = hashlib.md5(data + index_bytes).hexdigest()
    hash_int = int(full_hash[:3], 16) 
    return f"{hash_int:012b}"

def embed(image_path, output_path):
    print(f"Embedding: {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Image '{image_path}' not found.")
        return False
        
    h, w, c = img.shape
    h = (h // BLOCK_SIZE) * BLOCK_SIZE
    w = (w // BLOCK_SIZE) * BLOCK_SIZE
    img = img[:h, :w]
    
    watermarked_img = img.copy()
    total_blocks = (h // BLOCK_SIZE) * (w // BLOCK_SIZE)
    mapping = get_random_mapping(total_blocks, KEY)
    
    for channel_id in range(3):
        channel = watermarked_img[:, :, channel_id]
        blocks = []
        recovery_bits_list = []
        
        # --- PASS 1: Prepare Data ---
        for i in range(0, h, BLOCK_SIZE):
            for j in range(0, w, BLOCK_SIZE):
                block = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
                clean_block = (block & 0xFC) 
                blocks.append(clean_block)
                avg_val = int(np.mean(clean_block))
                recovery_bits_list.append(f"{avg_val:08b}")

        # --- PASS 2: Embed Data ---
        idx = 0
        for i in range(0, h, BLOCK_SIZE):
            for j in range(0, w, BLOCK_SIZE):
                current_block = blocks[idx]
                auth_payload = get_location_dependent_hash(current_block.flatten(), idx)
                partner_idx = mapping[idx]
                recovery_payload = recovery_bits_list[partner_idx]
                full_payload = auth_payload + recovery_payload
                
                flat = current_block.flatten()
                bit_idx = 0
                for k in range(10):
                    b1 = int(full_payload[bit_idx])
                    b2 = int(full_payload[bit_idx+1])
                    flat[k] = flat[k] | (b2 << 1) | b1
                    bit_idx += 2
                
                channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = flat.reshape(BLOCK_SIZE, BLOCK_SIZE)
                idx += 1
                
        watermarked_img[:, :, channel_id] = channel

    cv2.imwrite(output_path, watermarked_img)
    return True

def recover(image_path, output_path):
    print(f"Recovering: {image_path}...")
    img = cv2.imread(image_path)
    if img is None: return
    
    h, w, c = img.shape
    total_blocks = (h // BLOCK_SIZE) * (w // BLOCK_SIZE)
    mapping = get_random_mapping(total_blocks, KEY)
    
    reverse_mapping = np.zeros(total_blocks, dtype=int)
    for provider, receiver in enumerate(mapping):
        reverse_mapping[receiver] = provider

    recovered_img = img.copy()
    tamper_map = np.zeros((h, w), dtype=np.uint8)
    
    global_dead_mask = np.zeros((h, w), dtype=np.uint8)
    global_restored_mask = np.zeros((h, w), dtype=np.uint8)

    for channel_id in range(3):
        channel = img[:, :, channel_id]
        rec_channel = recovered_img[:, :, channel_id]
        
        extracted_auth = []
        extracted_recovery = []
        calculated_hashes = []
        
        # --- PASS 1: Analysis ---
        idx = 0
        tamper_count = 0
        
        for i in range(0, h, BLOCK_SIZE):
            for j in range(0, w, BLOCK_SIZE):
                block = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
                flat = block.flatten()
                
                payload = ""
                for k in range(10):
                    val = flat[k]
                    payload += str(val & 1) + str((val >> 1) & 1)
                
                extracted_auth.append(payload[:12])
                extracted_recovery.append(int(payload[12:], 2))
                
                clean_block = (block & 0xFC)
                cal_hash = get_location_dependent_hash(clean_block.flatten(), idx)
                calculated_hashes.append(cal_hash)
                
                if cal_hash != payload[:12]:
                    tamper_count += 1
                
                idx += 1

        # --- ADAPTIVE STRATEGY ---
        tamper_rate = tamper_count / total_blocks
        is_global_attack = tamper_rate > 0.40
        
        if is_global_attack and channel_id == 0:
            print(f"  - High Tamper Rate ({tamper_rate:.2f}). Adaptive Mode Engaged.")

        # --- PASS 2: Restoration ---
        idx = 0
        for i in range(0, h, BLOCK_SIZE):
            for j in range(0, w, BLOCK_SIZE):
                
                is_tampered = calculated_hashes[idx] != extracted_auth[idx]
                
                if is_tampered:
                    tamper_map[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = 255
                    block = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
                    block_mean = np.mean(block)
                    
                    # 1. LOCAL ATTACKS (Splicing, Copy-Move, Text)
                    if not is_global_attack:
                         provider_idx = reverse_mapping[idx]
                         if calculated_hashes[provider_idx] == extracted_auth[provider_idx]:
                             rec_channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = extracted_recovery[provider_idx]
                             global_restored_mask[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = 255
                         else:
                             global_dead_mask[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = 255

                    # 2. GLOBAL ATTACKS (Noise / Crop / JPEG)
                    else:
                        # Case A: Cropping (Block is destroyed/black) -> Use Backup
                        if block_mean < 5: 
                            provider_idx = reverse_mapping[idx]
                            if calculated_hashes[provider_idx] == extracted_auth[provider_idx]:
                                rec_channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = extracted_recovery[provider_idx]
                                global_restored_mask[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = 255
                            else:
                                global_dead_mask[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = 255
                        
                        # Case B: Noise (Pixel Repair) - HIGH CLARITY LOGIC
                        elif np.min(block) == 0 or np.max(block) == 255:
                            for py in range(BLOCK_SIZE):
                                for px in range(BLOCK_SIZE):
                                    pixel_val = block[py, px]
                                    # Identify Salt (255) or Pepper (0)
                                    if pixel_val == 0 or pixel_val == 255:
                                        y, x = i + py, j + px
                                        
                                        # PRIORITY: Smart Neighbor Median
                                        # (Preserves edges better than block average)
                                        neighbors = []
                                        for ny in range(max(0, y-1), min(h, y+2)):
                                            for nx in range(max(0, x-1), min(w, x+2)):
                                                val = channel[ny, nx]
                                                # Only accept CLEAN neighbors (not noise)
                                                if val > 0 and val < 255:
                                                    neighbors.append(val)
                                        
                                        if neighbors:
                                            # Sharpest repair: Median of clean neighbors
                                            rec_channel[y, x] = int(np.median(neighbors))
                                        else:
                                            # Fallback: If all neighbors are noisy (rare cluster)
                                            # THEN we use Backup (Block Average)
                                            provider_idx = reverse_mapping[idx]
                                            if calculated_hashes[provider_idx] == extracted_auth[provider_idx]:
                                                rec_channel[y, x] = extracted_recovery[provider_idx]
                                            else:
                                                # Extreme fallback: just median of what we have
                                                pass
                        
                        # Case C: JPEG -> Keep Original (Highest Quality)
                        else:
                            pass 

                idx += 1
        
        recovered_img[:, :, channel_id] = rec_channel

    # --- PASS 3: Final Polish ---
    
    # 1. Fill Dead Blocks (Holes)
    if np.sum(global_dead_mask) > 0:
         dead_dilated = cv2.dilate(global_dead_mask, np.ones((3,3), np.uint8), iterations=1)
         recovered_img = cv2.inpaint(recovered_img, dead_dilated, 5, cv2.INPAINT_TELEA)

    # 2. Refined Seam Blending (Tighter & Sharper)
    if np.sum(global_restored_mask) > 0:
        kernel = np.ones((3,3), np.uint8)
        seam_mask = cv2.morphologyEx(global_restored_mask, cv2.MORPH_GRADIENT, kernel)
        
        # REDUCED Dilation: 2 -> 1 (Thinner band for less blur)
        seam_mask = cv2.dilate(seam_mask, kernel, iterations=1) 
        
        # REDUCED Radius: 7 -> 3 (Sharper blend)
        recovered_img = cv2.inpaint(recovered_img, seam_mask, 3, cv2.INPAINT_TELEA)

    cv2.imwrite("final_tamper_map.png", tamper_map)
    cv2.imwrite(output_path, recovered_img)
    print(f"Success! Result saved to: {output_path}")

if __name__ == "__main__":
    pass