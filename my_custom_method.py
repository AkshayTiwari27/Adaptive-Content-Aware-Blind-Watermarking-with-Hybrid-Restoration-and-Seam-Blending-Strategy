import cv2
import numpy as np
import hashlib
import os

print("--- Smart Self-Recovery System (DLSBM v15 - Ultra-HD Recovery) ---")
print("--- [Fixed: Smart Center-Border Mapping + 2x2 High-Res Payload] ---")

# --- CONFIGURATION ---
BLOCK_SIZE = 4   # 4x4 Blocks
KEY = 9999       # Secret key

def get_smart_mapping(h, w, block_size, key):
    """
    Guarantees border blocks store backups in the center, and vice versa.
    This makes the system highly resistant to massive cropping attacks.
    """
    np.random.seed(key)
    blocks_y = h // block_size
    blocks_x = w // block_size
    total_blocks = blocks_y * blocks_x
    
    cy, cx = blocks_y / 2.0, blocks_x / 2.0
    
    distances = []
    idx = 0
    for i in range(blocks_y):
        for j in range(blocks_x):
            dist = (i - cy)**2 + (j - cx)**2
            distances.append((dist, idx))
            idx += 1
            
    # Sort by distance descending (Border blocks first, Center blocks last)
    distances.sort(key=lambda x: x[0], reverse=True)
    
    half = total_blocks // 2
    border_blocks = [x[1] for x in distances[:half]]
    center_blocks = [x[1] for x in distances[half:]]
    
    np.random.shuffle(border_blocks)
    np.random.shuffle(center_blocks)
    
    mapping = np.zeros(total_blocks, dtype=int)
    for i in range(half):
        b = border_blocks[i]
        c = center_blocks[i]
        mapping[b] = c  # Border block stores its backup in a Center block
        mapping[c] = b  # Center block stores its backup in a Border block
        
    # Handle odd number of total blocks (exact center maps to itself)
    if total_blocks % 2 != 0:
        mapping[center_blocks[-1]] = center_blocks[-1]
        
    return mapping

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
    mapping = get_smart_mapping(h, w, BLOCK_SIZE, KEY)
    
    for channel_id in range(3):
        channel = watermarked_img[:, :, channel_id]
        blocks = []
        recovery_bits_list = []
        
        # --- PASS 1: Prepare HQ Data (4 sub-blocks per 4x4 block) ---
        for i in range(0, h, BLOCK_SIZE):
            for j in range(0, w, BLOCK_SIZE):
                block = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
                clean_block = (block & 0xFC) 
                blocks.append(clean_block)
                
                # Calculate 2x2 sub-block averages for 4x higher resolution
                tl = np.mean(clean_block[0:2, 0:2])
                tr = np.mean(clean_block[0:2, 2:4])
                bl = np.mean(clean_block[2:4, 0:2])
                br = np.mean(clean_block[2:4, 2:4])
                
                # Quantize to 5 bits (0-31 range)
                q_tl, q_tr = int(tl) >> 3, int(tr) >> 3
                q_bl, q_br = int(bl) >> 3, int(br) >> 3
                
                # 5 bits * 4 quadrants = 20 bits
                payload = f"{q_tl:05b}{q_tr:05b}{q_bl:05b}{q_br:05b}"
                recovery_bits_list.append(payload)

        # --- PASS 2: Embed 32 Bits per block ---
        idx = 0
        for i in range(0, h, BLOCK_SIZE):
            for j in range(0, w, BLOCK_SIZE):
                current_block = blocks[idx]
                auth_payload = get_location_dependent_hash(current_block.flatten(), idx)
                
                partner_idx = mapping[idx]
                recovery_payload = recovery_bits_list[partner_idx]
                
                # 12 bits auth + 20 bits HQ recovery = 32 bits
                full_payload = auth_payload + recovery_payload 
                
                flat = current_block.flatten()
                bit_idx = 0
                # Embed across all 16 pixels using 2 LSBs (16 * 2 = 32)
                for k in range(16):
                    b1 = int(full_payload[bit_idx])
                    b2 = int(full_payload[bit_idx+1])
                    flat[k] = (flat[k] & 0xFC) | (b2 << 1) | b1
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
    mapping = get_smart_mapping(h, w, BLOCK_SIZE, KEY)
    
    # Reverse mapping logic is identical since mapping[b]=c and mapping[c]=b
    reverse_mapping = mapping 

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
        
        # --- PASS 1: Analysis & Extraction ---
        idx = 0
        tamper_count = 0
        
        for i in range(0, h, BLOCK_SIZE):
            for j in range(0, w, BLOCK_SIZE):
                block = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
                flat = block.flatten()
                
                payload = ""
                # Extract 32 bits from 16 pixels
                for k in range(16):
                    val = flat[k]
                    payload += str(val & 1) + str((val >> 1) & 1)
                
                extracted_auth.append(payload[:12])
                
                # Reconstruct the four 5-bit values back to 8-bit scale
                rec_tl = (int(payload[12:17], 2) << 3) + 4
                rec_tr = (int(payload[17:22], 2) << 3) + 4
                rec_bl = (int(payload[22:27], 2) << 3) + 4
                rec_br = (int(payload[27:32], 2) << 3) + 4
                
                extracted_recovery.append((rec_tl, rec_tr, rec_bl, rec_br))
                
                clean_block = (block & 0xFC)
                cal_hash = get_location_dependent_hash(clean_block.flatten(), idx)
                calculated_hashes.append(cal_hash)
                
                if cal_hash != payload[:12]:
                    tamper_count += 1
                
                idx += 1

        tamper_rate = tamper_count / ( (h // BLOCK_SIZE) * (w // BLOCK_SIZE) )
        is_global_attack = tamper_rate > 0.40

        # --- PASS 2: Restoration ---
        idx = 0
        for i in range(0, h, BLOCK_SIZE):
            for j in range(0, w, BLOCK_SIZE):
                
                is_tampered = calculated_hashes[idx] != extracted_auth[idx]
                
                if is_tampered:
                    tamper_map[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = 255
                    block = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
                    block_mean = np.mean(block)
                    provider_idx = reverse_mapping[idx]
                    is_backup_valid = calculated_hashes[provider_idx] == extracted_auth[provider_idx]
                    
                    # 1. LOCAL ATTACKS & CROPPING (Block is destroyed/black)
                    if not is_global_attack or block_mean < 5:
                        if is_backup_valid:
                            hq_vals = extracted_recovery[provider_idx]
                            # Apply the high-res 2x2 quadrants
                            rec_channel[i:i+2, j:j+2] = hq_vals[0]
                            rec_channel[i:i+2, j+2:j+4] = hq_vals[1]
                            rec_channel[i+2:i+4, j:j+2] = hq_vals[2]
                            rec_channel[i+2:i+4, j+2:j+4] = hq_vals[3]
                            global_restored_mask[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = 255
                        else:
                            global_dead_mask[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = 255

                    # 2. NOISE (Pixel Repair)
                    elif is_global_attack and (np.min(block) == 0 or np.max(block) == 255):
                        for py in range(BLOCK_SIZE):
                            for px in range(BLOCK_SIZE):
                                pixel_val = block[py, px]
                                if pixel_val == 0 or pixel_val == 255:
                                    y, x = i + py, j + px
                                    
                                    neighbors = []
                                    for ny in range(max(0, y-1), min(h, y+2)):
                                        for nx in range(max(0, x-1), min(w, x+2)):
                                            val = channel[ny, nx]
                                            if val > 0 and val < 255:
                                                neighbors.append(val)
                                    
                                    if neighbors:
                                        rec_channel[y, x] = int(np.median(neighbors))
                                    elif is_backup_valid:
                                        # Map specific pixel to its proper 2x2 quadrant backup
                                        quad_idx = (py // 2) * 2 + (px // 2)
                                        rec_channel[y, x] = extracted_recovery[provider_idx][quad_idx]

                idx += 1
        
        recovered_img[:, :, channel_id] = rec_channel

    # --- PASS 3: Final Polish ---
    if np.sum(global_dead_mask) > 0:
         dead_dilated = cv2.dilate(global_dead_mask, np.ones((3,3), np.uint8), iterations=1)
         recovered_img = cv2.inpaint(recovered_img, dead_dilated, 5, cv2.INPAINT_TELEA)

    if np.sum(global_restored_mask) > 0:
        kernel = np.ones((3,3), np.uint8)
        seam_mask = cv2.morphologyEx(global_restored_mask, cv2.MORPH_GRADIENT, kernel)
        seam_mask = cv2.dilate(seam_mask, kernel, iterations=1) 
        # Using a tiny radius ensures we keep the high-res detail while blending the edges
        recovered_img = cv2.inpaint(recovered_img, seam_mask, 2, cv2.INPAINT_TELEA)

    cv2.imwrite("final_tamper_map.png", tamper_map)
    cv2.imwrite(output_path, recovered_img)
    print(f"Success! High-Res Result saved to: {output_path}")

if __name__ == "__main__":
    pass