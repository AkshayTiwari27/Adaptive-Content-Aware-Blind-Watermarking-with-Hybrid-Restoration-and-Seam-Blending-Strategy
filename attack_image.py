import cv2
import numpy as np
import os

# --- Attack Function Definitions ---

def attack_content_removal(image, x=220, y=220, w=100, h=80):
    """(Forgery Type: Erasing) Blacks out a rectangular region."""
    attacked_image = image.copy()
    cv2.rectangle(attacked_image, (x, y), (x + w, y + h), (0, 0, 0), -1)
    return attacked_image, 'Content Removal'

def attack_copy_move(image, src_x=None, src_y=None, w=None, h=None, dst_x=None, dst_y=None):
    """(Forgery Type: Cloning) Copies a region to another location."""
    attacked_image = image.copy()
    rows, cols = attacked_image.shape[:2]

    # Defaults: 15% size, Source=Bottom-Right, Dest=Top-Left
    if w is None: w = int(cols * 0.15)
    if h is None: h = int(rows * 0.15)
    if src_x is None: src_x = int(cols * 0.6)
    if src_y is None: src_y = int(rows * 0.6)
    if dst_x is None: dst_x = int(cols * 0.1)
    if dst_y is None: dst_y = int(rows * 0.1)

    # Bounds Check
    if src_x + w > cols: src_x = cols - w
    if src_y + h > rows: src_y = rows - h
    if dst_x + w > cols: dst_x = cols - w
    if dst_y + h > rows: dst_y = rows - h

    source_region = attacked_image[src_y:src_y+h, src_x:src_x+w]
    attacked_image[dst_y:dst_y+h, dst_x:dst_x+w] = source_region
    
    return attacked_image, 'Copy-Move'

def attack_political_splicing(image, face_image_path='politician_face.tiff', x=50, y=50, w=100, h=100):
    """(Forgery Type: Splicing) Pastes an external image."""
    attacked_image = image.copy()
    rows, cols = attacked_image.shape[:2]
    
    try:
        face_image = cv2.imread(face_image_path)
        if face_image is None: raise FileNotFoundError

        if w >= cols or h >= rows:
            w, h = int(cols / 3), int(rows / 3)
        if x + w > cols: x = cols - w
        if y + h > rows: y = rows - h

        face_resized = cv2.resize(face_image, (w, h))
        attacked_image[y:y+h, x:x+w] = face_resized
        return attacked_image, 'Political Splicing'

    except FileNotFoundError:
        print(f"  - ERROR: Splice image '{face_image_path}' not found.")
        return None, None

def attack_cropping(image, percent=40):
    """
    (Forgery Type: Cropping)
    Paper Standard: Keeps the central region and zeroes out the BORDER.
    The 'percent' parameter is the percentage of area REMOVED (border).
    Does NOT resize, as resizing destroys the block grid.
    """
    attacked_image = image.copy()
    h, w = attacked_image.shape[:2]
    
    # Calculate the dimensions of the KEPT central region
    # If percent=40, we remove 40% of area, keeping 60%
    keep_ratio = np.sqrt(1.0 - percent / 100.0)
    keep_h = int(h * keep_ratio)
    keep_w = int(w * keep_ratio)
    
    # Align to block size (4) to avoid partial block issues
    keep_h = (keep_h // 4) * 4
    keep_w = (keep_w // 4) * 4
    
    # Center coordinates of the kept region
    start_y = (h - keep_h) // 2
    start_x = (w - keep_w) // 2
    
    # Zero out the border, keep the center
    center = attacked_image[start_y:start_y+keep_h, start_x:start_x+keep_w].copy()
    attacked_image[:] = 0
    attacked_image[start_y:start_y+keep_h, start_x:start_x+keep_w] = center
    
    return attacked_image, f'Cropping ({percent}%)'

def attack_jpeg_compression(image, quality=90):
    """
    (Forgery Type: JPEG)
    Uses Higher Quality (90) to allow some LSBs to survive.
    """
    temp_filename = 'temp_compressed.jpg'
    cv2.imwrite(temp_filename, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    attacked_image = cv2.imread(temp_filename)
    
    # Ensure it's read back in the same mode (grayscale)
    if len(image.shape) == 2 and len(attacked_image.shape) == 3:
        attacked_image = cv2.cvtColor(attacked_image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and len(attacked_image.shape) == 3:
        # If input was color but saved/loaded, ensure consistency if needed
        pass

    os.remove(temp_filename)
    return attacked_image, f'JPEG Compression (Q{quality})'

def attack_salt_and_pepper(image, amount=0.05):
    """
    (Forgery Type: Noise)
    Adds Salt (255) and Pepper (0) noise.
    """
    attacked_image = image.copy()
    
    # Handle both Grayscale (2D) and Color (3D) images
    if len(image.shape) == 2:
        row, col = image.shape
        num_pixels = int(amount * (row * col))
        
        # Salt
        for _ in range(num_pixels // 2):
            y, x = np.random.randint(0, row), np.random.randint(0, col)
            attacked_image[y, x] = 255
        # Pepper
        for _ in range(num_pixels // 2):
            y, x = np.random.randint(0, row), np.random.randint(0, col)
            attacked_image[y, x] = 0
            
    else:
        row, col, ch = image.shape
        num_pixels = int(amount * (row * col))
        
        # For color, we usually apply noise to all channels at the same pixel
        # Salt
        for _ in range(num_pixels // 2):
            y, x = np.random.randint(0, row), np.random.randint(0, col)
            attacked_image[y, x] = (255, 255, 255)
        # Pepper
        for _ in range(num_pixels // 2):
            y, x = np.random.randint(0, row), np.random.randint(0, col)
            attacked_image[y, x] = (0, 0, 0)
        
    return attacked_image, f'Salt & Pepper Noise ({int(amount*100)}%)'