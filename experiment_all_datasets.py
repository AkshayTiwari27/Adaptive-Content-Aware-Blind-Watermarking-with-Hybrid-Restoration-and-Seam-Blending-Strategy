"""
experiment_all_datasets.py
==============================================
Runs 5 attacks on 5 representative images from each of 5 datasets:
    Kodak, BSDS300, Set14, BOSSBase, UCID

Attacks:
    content_removal  - blackout 100x80 region at default position
    copy_move        - src=top-left border, dst=centre (30% area, block-aligned)
                       Mirrors generate_varying_tamper_grids.py — border backups
                       stay in untouched border blocks so recovery is clean.
    splicing         - adjacent dataset image pasted at centre (30% area)
    jpeg_q70         - JPEG compression at quality=70
    snp_005          - salt-and-pepper noise at density=5%

Output:
    dataset_results/
        {Dataset}/
            {image_name}/
                orig.png, wmk.png
                {attack}_atk.png, {attack}_rec.png, {attack}_map.png
        summary.csv
"""

import os
import sys
import io
import csv
import math
import shutil
import zipfile
import urllib.request

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

import my_custom_method as wm
import attack_image as attacker

# Force UTF-8 on Windows to avoid cp1252 encode errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
RESULTS_DIR = "dataset_results"
CACHE_DIR   = "_image_cache"
IMG_SIZE    = 512

BOSS_ZIP = r"C:\Users\tiwar\Downloads\bossbase.zip"
UCID_ZIP = r"C:\Users\tiwar\Downloads\ucid.zip"

# 5 representative images per dataset
DATASETS = {
    "Kodak": {
        "type": "url",
        "images": {
            "kodim04": "http://r0k.us/graphics/kodak/kodak/kodim04.png",
            "kodim07": "http://r0k.us/graphics/kodak/kodak/kodim07.png",
            "kodim13": "http://r0k.us/graphics/kodak/kodak/kodim13.png",
            "kodim15": "http://r0k.us/graphics/kodak/kodak/kodim15.png",
            "kodim23": "http://r0k.us/graphics/kodak/kodak/kodim23.png",
        }
    },
    "BSDS300": {
        "type": "url",
        "images": {
            "bsds_101085": "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/images/test/101085.jpg",
            "bsds_101087": "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/images/test/101087.jpg",
            "bsds_105025": "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/images/test/105025.jpg",
            "bsds_119082": "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/images/test/119082.jpg",
            "bsds_126007": "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/images/test/126007.jpg",
        }
    },
    "Set14": {
        "type": "url",
        "images": {
            "set14_01": "https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_001_SRF_4_HR.png",
            "set14_06": "https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_006_SRF_4_HR.png",
            "set14_09": "https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_009_SRF_4_HR.png",
            "set14_12": "https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_012_SRF_4_HR.png",
            "set14_14": "https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_014_SRF_4_HR.png",
        }
    },
    "BOSSBase": {
        "type": "zip",
        "zip_path": BOSS_ZIP,
        "images": {
            "boss_0001": "grayscale/QF95/1.jpg",
            "boss_1000": "grayscale/QF95/1000.jpg",
            "boss_3000": "grayscale/QF95/3000.jpg",
            "boss_5000": "grayscale/QF95/5000.jpg",
            "boss_7000": "grayscale/QF95/7000.jpg",
        }
    },
    "UCID": {
        "type": "zip",
        "zip_path": UCID_ZIP,
        "images": {
            "ucid_0001": "UCID1338/1.tif",
            "ucid_0100": "UCID1338/100.tif",
            "ucid_0300": "UCID1338/300.tif",
            "ucid_0500": "UCID1338/500.tif",
            "ucid_0700": "UCID1338/700.tif",
        }
    },
}

ATTACK_KEYS = ["content_removal", "copy_move", "splicing", "jpeg_q70", "snp_005"]


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def load_from_url(url, name):
    """Download from URL with local cache. Returns BGR ndarray or None."""
    cache_path = os.path.join(CACHE_DIR, name + ".png")
    if os.path.exists(cache_path):
        img = cv2.imread(cache_path)
        if img is not None:
            return img
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=30).read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imwrite(cache_path, img)
        return img
    except Exception as e:
        print(f"  WARN: could not download {name}: {e}")
        return None


def load_from_zip(zip_path, zip_entry, name):
    """Load image from inside a zip file. Returns BGR ndarray or None."""
    cache_path = os.path.join(CACHE_DIR, name + ".png")
    if os.path.exists(cache_path):
        img = cv2.imread(cache_path)
        if img is not None:
            return img
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            data = zf.read(zip_entry)
        if zip_entry.lower().endswith((".tif", ".tiff")):
            # PIL handles TIFF reliably; cv2 may not
            pil_img = Image.open(io.BytesIO(data)).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imwrite(cache_path, img)
        return img
    except Exception as e:
        print(f"  WARN: could not load {name} from zip: {e}")
        return None


def load_dataset_images(ds_name, ds_cfg):
    """Load all 5 images for a dataset. Returns ordered dict name->BGR."""
    images = {}
    for img_name, src in ds_cfg["images"].items():
        if ds_cfg["type"] == "url":
            img = load_from_url(src, img_name)
        else:
            img = load_from_zip(ds_cfg["zip_path"], src, img_name)
        if img is None:
            print(f"  SKIP: {img_name} (load failed)")
            continue
        # Ensure 3-channel BGR
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Resize to 512x512
        if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_LANCZOS4)
        # Clip to [4, 251] so watermarked pixels can never be exactly 0 or 255.
        # After 2-LSB embedding: P' = (P & 0xFC) | bits.
        # P >=  4  →  P' >=  4 (no zero pixels)
        # P <= 251  →  P' <= 251 (no 255 pixels)
        # This prevents SCBM's noise_ratio heuristic from misfiring on
        # copy-move / splicing attacks against naturally dark/bright images
        # (mirrors the preprocessing applied to USC-SIPI grayscale_normalized/).
        img = np.clip(img.astype(np.int16), 4, 251).astype(np.uint8)
        images[img_name] = img
    return images


# ---------------------------------------------------------------------------
# Centre-region helper (mirrors generate_varying_tamper_grids.py)
# ---------------------------------------------------------------------------
def _centre_region(img, percent=20):
    """Return (x, y, w, h) for a block-aligned centred region at `percent` area.

    Matches the formula used in generate_varying_tamper_grids.py.  At 20%,
    the destination corners sit at block distance ~41 from centre — safely
    inside the SCBM centre zone (radius ~51 blocks for 512x512).  All
    destination blocks are centre blocks whose backups live in untouched
    border blocks, guaranteeing clean recovery.
    """
    rows, cols = img.shape[:2]
    ratio = math.sqrt(percent / 100.0)
    w = (int(cols * ratio) // 4) * 4
    h = (int(rows * ratio) // 4) * 4
    x = ((cols - w) // 2 // 4) * 4
    y = ((rows - h) // 2 // 4) * 4
    return x, y, w, h


# ---------------------------------------------------------------------------
# Copy-move attack: src=top-left (border zone), dst=centre
# ---------------------------------------------------------------------------
def attack_copy_move_centre(img):
    """Copy top-left region to the image centre (30% area, block-aligned).

    Source is from the top-left border zone; destination is the image centre.
    Centre-block backups live in border blocks → border stays untouched →
    SCBM can fully reconstruct the centre region.
    """
    x, y, w, h = _centre_region(img)
    out = img.copy()
    src_patch = out[0:h, 0:w].copy()   # top-left corner (border zone)
    out[y:y + h, x:x + w] = src_patch  # paste to centre
    return out, "Copy-Move"


# ---------------------------------------------------------------------------
# Splicing attack: donor patch pasted at centre
# ---------------------------------------------------------------------------
def attack_splicing_centre(target_img, donor_img):
    """Paste a large patch from donor's centre into target's centre (30% area).

    Same centre-paste logic as copy-move: centre backup is in border blocks
    (untouched), so SCBM recovers cleanly.
    """
    x, y, w, h = _centre_region(target_img)
    out = target_img.copy()
    dh, dw = donor_img.shape[:2]
    # Crop donor patch from its own centre
    py = max(0, min(dh // 2 - h // 2, dh - h))
    px = max(0, min(dw // 2 - w // 2, dw - w))
    patch = donor_img[py:py + h, px:px + w]
    # Harmonise channel count
    if out.ndim == 2 and patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    elif out.ndim == 3 and patch.ndim == 2:
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    out[y:y + h, x:x + w] = patch
    return out, "Splicing"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(img1, img2):
    if img1.ndim == 3:
        return ssim_fn(img1, img2, channel_axis=2, data_range=255)
    return ssim_fn(img1, img2, data_range=255)


# ---------------------------------------------------------------------------
# Process one dataset
# ---------------------------------------------------------------------------
def run_dataset(ds_name, ds_cfg, csv_rows):
    print(f"\n{'='*60}")
    print(f"  DATASET: {ds_name}")
    print(f"{'='*60}")

    raw_images = load_dataset_images(ds_name, ds_cfg)
    if not raw_images:
        print("  ERROR: no images loaded, skipping dataset.")
        return

    names = list(raw_images.keys())

    for i, img_name in enumerate(names):
        orig_bgr  = raw_images[img_name]
        # Adjacent image (circular) used as splice donor
        donor_bgr = raw_images[names[(i + 1) % len(names)]]

        img_dir = os.path.join(RESULTS_DIR, ds_name, img_name)
        os.makedirs(img_dir, exist_ok=True)

        # Save original
        orig_path = os.path.join(img_dir, "orig.png")
        cv2.imwrite(orig_path, orig_bgr)

        # Embed watermark
        wmk_path = os.path.join(img_dir, "wmk.png")
        print(f"\n  [{img_name}] embedding watermark ...")
        ok = wm.embed(orig_path, wmk_path)
        if not ok or not os.path.exists(wmk_path):
            print(f"  ERROR: embed failed for {img_name}, skipping.")
            continue

        wmk_bgr = cv2.imread(wmk_path)

        # --- Apply each attack, recover, copy tamper map ---
        for atk_key in ATTACK_KEYS:
            atk_path = os.path.join(img_dir, f"{atk_key}_atk.png")
            rec_path = os.path.join(img_dir, f"{atk_key}_rec.png")
            map_path = os.path.join(img_dir, f"{atk_key}_map.png")

            if atk_key == "content_removal":
                attacked, _ = attacker.attack_content_removal(wmk_bgr)
            elif atk_key == "copy_move":
                # src=top-left border, dst=centre (mirrors generate_varying_tamper_grids.py)
                attacked, _ = attack_copy_move_centre(wmk_bgr)
            elif atk_key == "splicing":
                # donor patch pasted at centre — same SCBM-safe zone as copy-move
                attacked, _ = attack_splicing_centre(wmk_bgr, donor_bgr)
            elif atk_key == "jpeg_q70":
                attacked, _ = attacker.attack_jpeg_compression(wmk_bgr, quality=70)
            elif atk_key == "snp_005":
                attacked, _ = attacker.attack_salt_and_pepper(wmk_bgr, amount=0.05)

            cv2.imwrite(atk_path, attacked)

            wm.recover(atk_path, rec_path)

            # Copy tamper map IMMEDIATELY — recover() writes final_tamper_map.png
            hardcoded_map = "final_tamper_map.png"
            if os.path.exists(hardcoded_map):
                shutil.copy2(hardcoded_map, map_path)
            else:
                h, w_px = wmk_bgr.shape[:2]
                cv2.imwrite(map_path, np.zeros((h, w_px), dtype=np.uint8))

            # Metrics
            orig_read = cv2.imread(orig_path)
            rec_read  = cv2.imread(rec_path)
            if rec_read is not None and orig_read is not None:
                p = compute_psnr(orig_read, rec_read)
                s = compute_ssim(orig_read, rec_read)
                print(f"  [{img_name}] {atk_key:18s}  PSNR={p:6.2f} dB  SSIM={s:.4f}")
                csv_rows.append({
                    "dataset": ds_name,
                    "image":   img_name,
                    "attack":  atk_key,
                    "psnr":    f"{p:.4f}",
                    "ssim":    f"{s:.6f}",
                })
            else:
                print(f"  WARN: could not read recovered image for {img_name}/{atk_key}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_rows = []

    for ds_name, ds_cfg in DATASETS.items():
        run_dataset(ds_name, ds_cfg, csv_rows)

    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "image", "attack", "psnr", "ssim"]
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n\nAll experiments done.")
    print(f"Results : {os.path.abspath(RESULTS_DIR)}/")
    print(f"Summary : {os.path.abspath(csv_path)}")


if __name__ == "__main__":
    main()
