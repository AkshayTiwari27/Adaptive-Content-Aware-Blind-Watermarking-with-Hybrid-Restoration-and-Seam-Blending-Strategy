"""
ablation_study.py
==================
Addresses two reviewer comments with quantitative ablation experiments.

Experiment A – Module Ablation (Comment 1)
  A1. SCBM mapping vs. Random mapping vs. Sequential mapping
      → Tests on cropping and content-removal attacks across all 9 images
      → Shows SCBM's strategic center-border placement is critical

  A2. 3-Way Classifier variants
      A2a. No classifier   — always Branch C (structural recovery)
      A2b. 2-way, no Br-A  — if η>0.5% → B, else C  (no JPEG bypass)
      A2c. 2-way, no Br-B  — if τ>85% → A, else C   (no noise repair)
      A2d. Full DLSBM      — 3-way A/B/C (current)
      → Tests on all 6 standard attack types

Experiment B – Threshold Sensitivity (Comment 2)
  Sweep τ_th ∈ {60,65,70,75,80,85,90,95} %
  Sweep η_th ∈ {0.1, 0.3, 0.5, 1.0, 2.0, 5.0} %
  Metric: correct branch classification rate across all 9 images × 6 attacks

Outputs (all in ablation_results/)
-------
  Table_SCBM_Ablation.csv         — Exp A1
  Table_Classifier_Ablation.csv   — Exp A2
  Table_Threshold_Sweep.csv       — Exp B (heatmap data)
  Table_Threshold_PSNR.csv        — R-PSNR at boundary threshold values

Run from:  C:/Users/tiwar/Downloads/journal implementation/
"""

import cv2, os, glob, math, csv, hashlib, sys
import numpy as np
from scipy import stats
from skimage.metrics import structural_similarity as _ssim
import attack_image as attacker

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError: pass

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_DIR  = "grayscale_normalized"
OUT_DIR    = "ablation_results"
BLOCK_SIZE = 4
KEY        = 9999
np.random.seed(42)

# ── Metric helpers ────────────────────────────────────────────────────────────
def psnr(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    mse = np.mean((a - b) ** 2)
    return 100.0 if mse < 1e-12 else 20 * math.log10(255.0 / math.sqrt(mse))

def ssim_v(a, b):
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    ax = 2 if a.ndim == 3 else None
    return float(_ssim(a, b, data_range=255, channel_axis=ax))

def ms_ssim(a, b, w=(0.0448,0.2856,0.3001,0.2363,0.1333)):
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    sc, sw = [], []
    ax = 2 if a.ndim == 3 else None
    for wt in w:
        if a.shape[0] < 16 or a.shape[1] < 16: break
        sc.append(_ssim(a, b, data_range=255, channel_axis=ax)); sw.append(wt)
        a = cv2.resize(a,(a.shape[1]//2, a.shape[0]//2))
        b = cv2.resize(b,(b.shape[1]//2, b.shape[0]//2))
    if not sc: return 0.0
    t = sum(sw)
    return float(np.prod([s**(wv/t) for s,wv in zip(sc,sw)]))

def ci95(v):
    n=len(v); m=float(np.mean(v)); s=float(np.std(v,ddof=1))
    return m, s, float(stats.t.ppf(0.975,df=n-1)*s/math.sqrt(n))

# ── Location-dependent hash (matches my_custom_method.py) ────────────────────
def loc_hash(flat_block, idx):
    h = hashlib.md5(flat_block.tobytes()+int(idx).to_bytes(4,'big')).hexdigest()
    return f"{int(h[:3],16):012b}"

# ══════════════════════════════════════════════════════════════════════════════
# MAPPING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def mapping_scbm(h, w, bs, key):
    """Current DLSBM: border blocks backup in center blocks."""
    np.random.seed(key)
    by, bx = h//bs, w//bs
    total  = by * bx
    cy, cx = by/2.0, bx/2.0
    dists  = sorted([(( i-cy)**2+(j-cx)**2, i*bx+j)
                     for i in range(by) for j in range(bx)], reverse=True)
    half   = total // 2
    border = [d[1] for d in dists[:half]]
    center = [d[1] for d in dists[half:]]
    np.random.shuffle(border); np.random.shuffle(center)
    m = np.zeros(total, dtype=int)
    for i in range(half):
        m[border[i]] = center[i]; m[center[i]] = border[i]
    if total % 2 != 0: m[center[-1]] = center[-1]
    return m

def mapping_random(h, w, bs, key):
    """Random bijective mapping — no spatial bias."""
    np.random.seed(key + 1)   # different seed so it's genuinely different
    total = (h//bs)*(w//bs)
    perm  = np.random.permutation(total)
    m     = np.zeros(total, dtype=int)
    for i in range(0, total-1, 2):
        m[perm[i]] = perm[i+1]; m[perm[i+1]] = perm[i]
    if total % 2 != 0: m[perm[-1]] = perm[-1]
    return m

def mapping_sequential(h, w, bs, key):
    """Sequential shift: block i backs up in block (i+N//2) % N."""
    total = (h//bs)*(w//bs)
    half  = total // 2
    m     = np.zeros(total, dtype=int)
    for i in range(total):
        m[i] = (i + half) % total
    return m

# ══════════════════════════════════════════════════════════════════════════════
# EMBED / RECOVER with swappable mapping and branch override
# ══════════════════════════════════════════════════════════════════════════════
def embed_img(img_bgr, mapping_fn):
    """Embed watermark using the supplied mapping function."""
    img = img_bgr.copy()
    h, w = img.shape[:2]
    h = (h // BLOCK_SIZE)*BLOCK_SIZE; w = (w // BLOCK_SIZE)*BLOCK_SIZE
    img = img[:h, :w]
    mapping = mapping_fn(h, w, BLOCK_SIZE, KEY)
    out = img.copy()

    for ch in range(3):
        channel = out[:, :, ch]
        blocks, rec_bits = [], []
        for i in range(0, h, BLOCK_SIZE):
            for j in range(0, w, BLOCK_SIZE):
                blk = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
                cb  = blk & 0xFC
                blocks.append(cb)
                tl=np.mean(cb[0:2,0:2]); tr=np.mean(cb[0:2,2:4])
                bl=np.mean(cb[2:4,0:2]); br=np.mean(cb[2:4,2:4])
                rec_bits.append(f"{int(tl)>>3:05b}{int(tr)>>3:05b}"
                                f"{int(bl)>>3:05b}{int(br)>>3:05b}")
        idx = 0
        for i in range(0, h, BLOCK_SIZE):
            for j in range(0, w, BLOCK_SIZE):
                cb   = blocks[idx]
                auth = loc_hash(cb.flatten(), idx)
                ptnr = mapping[idx]
                payload = auth + rec_bits[ptnr]
                flat = cb.flatten(); bi = 0
                for k in range(16):
                    b1=int(payload[bi]); b2=int(payload[bi+1])
                    flat[k] = (flat[k] & 0xFC) | (b2<<1) | b1
                    bi += 2
                channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = flat.reshape(BLOCK_SIZE, BLOCK_SIZE)
                idx += 1
        out[:, :, ch] = channel
    return out

def recover_img(img_bgr, mapping_fn, branch_override=None):
    """
    Recover image using supplied mapping and optional branch override.

    branch_override:
      None    → auto 3-way classification (full DLSBM)
      'noA'   → disable Branch A: η>0.5% → B, else C
      'noB'   → disable Branch B: τ>85% → A, else C
      'C'     → always Branch C (no classifier)
    """
    img = img_bgr.copy()
    h, w = img.shape[:2]
    h = (h//BLOCK_SIZE)*BLOCK_SIZE; w = (w//BLOCK_SIZE)*BLOCK_SIZE
    img = img[:h, :w]
    mapping = mapping_fn(h, w, BLOCK_SIZE, KEY)
    recovered = img.copy()
    dead_mask     = np.zeros((h,w), dtype=np.uint8)
    restored_mask = np.zeros((h,w), dtype=np.uint8)

    for ch in range(3):
        channel   = img[:, :, ch]
        rec_ch    = recovered[:, :, ch]
        ext_auth, ext_rec, cal_hash = [], [], []
        tamper_cnt = 0
        total_blk  = (h//BLOCK_SIZE)*(w//BLOCK_SIZE)

        for i in range(0, h, BLOCK_SIZE):
            for j in range(0, w, BLOCK_SIZE):
                blk  = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
                flat = blk.flatten()
                bits = "".join(str((v>>b)&1) for v in flat for b in [0,1])
                ext_auth.append(bits[:12])
                rec_tl=(int(bits[12:17],2)<<3)+4; rec_tr=(int(bits[17:22],2)<<3)+4
                rec_bl=(int(bits[22:27],2)<<3)+4; rec_br=(int(bits[27:32],2)<<3)+4
                ext_rec.append((rec_tl,rec_tr,rec_bl,rec_br))
                cb = (blk & 0xFC)
                ch_hash = loc_hash(cb.flatten(), len(cal_hash))
                cal_hash.append(ch_hash)
                if ch_hash != bits[:12]: tamper_cnt += 1

        tau = tamper_cnt / total_blk
        eta = np.sum((channel==0)|(channel==255)) / (h*w)

        # --- branch selection ---
        if branch_override == 'C':
            is_jpeg = False; is_noise = False
        elif branch_override == 'noA':
            is_noise = eta > 0.005; is_jpeg = False
        elif branch_override == 'noB':
            is_noise = False; is_jpeg = tau > 0.85
        else:   # full 3-way
            is_noise = eta > 0.005
            is_jpeg  = tau > 0.85 and not is_noise

        idx = 0
        for i in range(0, h, BLOCK_SIZE):
            for j in range(0, w, BLOCK_SIZE):
                is_tampered = cal_hash[idx] != ext_auth[idx]
                if is_tampered:
                    blk        = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
                    blk_mean   = np.mean(blk)
                    ptnr       = mapping[idx]
                    backup_ok  = cal_hash[ptnr] == ext_auth[ptnr]

                    if is_jpeg and blk_mean > 5:
                        pass  # Branch A: bypass

                    elif not is_noise or blk_mean < 5:
                        if backup_ok:
                            q = ext_rec[ptnr]
                            rec_ch[i:i+2,j:j+2]=q[0]; rec_ch[i:i+2,j+2:j+4]=q[1]
                            rec_ch[i+2:i+4,j:j+2]=q[2]; rec_ch[i+2:i+4,j+2:j+4]=q[3]
                            restored_mask[i:i+BLOCK_SIZE,j:j+BLOCK_SIZE]=255
                        else:
                            dead_mask[i:i+BLOCK_SIZE,j:j+BLOCK_SIZE]=255

                    elif is_noise:
                        for py in range(BLOCK_SIZE):
                            for px in range(BLOCK_SIZE):
                                pv = blk[py,px]
                                if pv == 0 or pv == 255:
                                    y,x = i+py, j+px
                                    nbrs=[channel[ny,nx]
                                          for ny in range(max(0,y-1),min(h,y+2))
                                          for nx in range(max(0,x-1),min(w,x+2))
                                          if 0<channel[ny,nx]<255]
                                    if nbrs:
                                        rec_ch[y,x]=int(np.median(nbrs))
                                    elif backup_ok:
                                        qi=(py//2)*2+(px//2)
                                        rec_ch[y,x]=ext_rec[ptnr][qi]
                idx += 1
        recovered[:, :, ch] = rec_ch

    if np.sum(dead_mask) > 0:
        d2 = cv2.dilate(dead_mask, np.ones((3,3),np.uint8), iterations=1)
        recovered = cv2.inpaint(recovered, d2, 5, cv2.INPAINT_TELEA)
    if np.sum(restored_mask) > 0:
        k  = np.ones((3,3),np.uint8)
        sm = cv2.morphologyEx(restored_mask, cv2.MORPH_GRADIENT, k)
        sm = cv2.dilate(sm, k, iterations=1)
        recovered = cv2.inpaint(recovered, sm, 2, cv2.INPAINT_TELEA)
    return recovered

# ══════════════════════════════════════════════════════════════════════════════
# ATTACK HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def do_attack(wm, name):
    if name == "content_removal":  return attacker.attack_content_removal(wm)[0]
    if name == "copy_move":        return attacker.attack_copy_move(wm)[0]
    if name == "splicing":
        atk, _ = attacker.attack_political_splicing(wm)
        return atk
    if name == "jpeg":             return attacker.attack_jpeg_compression(wm, 90)[0]
    if name == "noise":            return attacker.attack_salt_and_pepper(wm, 0.05)[0]
    if name == "cropping":         return attacker.attack_cropping(wm, 40)[0]
    return None

ATTACKS = ["content_removal","copy_move","splicing","jpeg","noise","cropping"]
ATTACK_LABELS = {
    "content_removal": "Content Removal",
    "copy_move":       "Copy-Move",
    "splicing":        "Splicing",
    "jpeg":            "JPEG ($Q=90$)",
    "noise":           "S\\&P (0.05)",
    "cropping":        "Cropping (40\\%)",
}

# Ground truth optimal branch for each attack (determines correct classification)
OPTIMAL_BRANCH = {
    "content_removal": "C",
    "copy_move":       "C",
    "splicing":        "C",
    "jpeg":            "A",
    "noise":           "B",
    "cropping":        "C",   # zeroed blocks use block_mean<5 override regardless
}

def get_files():
    files = []
    for ext in ("*.tiff","*.png","*.jpg","*.jpeg"):
        files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    return sorted(files)

def classify(img, tau_th=0.85, eta_th=0.005):
    """Return (tau, eta, branch) using given thresholds."""
    img = img.copy()
    h,w = img.shape[:2]
    h=(h//BLOCK_SIZE)*BLOCK_SIZE; w=(w//BLOCK_SIZE)*BLOCK_SIZE
    img=img[:h,:w]
    total=(h//BLOCK_SIZE)*(w//BLOCK_SIZE)
    tampered=0
    for ch in range(3):
        ch_d=img[:,:,ch]; idx=0
        for i in range(0,h,BLOCK_SIZE):
            for j in range(0,w,BLOCK_SIZE):
                blk=ch_d[i:i+BLOCK_SIZE,j:j+BLOCK_SIZE].flatten()
                bits="".join(str((v>>b)&1) for v in blk for b in [0,1])
                cb=(ch_d[i:i+BLOCK_SIZE,j:j+BLOCK_SIZE]&0xFC).flatten()
                if loc_hash(cb,idx)!=bits[:12]: tampered+=1
                idx+=1
    tau=tampered/(total*3)
    eta=np.sum((img==0)|(img==255))/(h*w*3)
    is_noise=eta>eta_th
    is_jpeg=tau>tau_th and not is_noise
    br="A" if is_jpeg else ("B" if is_noise else "C")
    return tau, eta, br

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A1 — SCBM MAPPING ABLATION
# ══════════════════════════════════════════════════════════════════════════════
def exp_A1_scbm(files):
    print("\n" + "="*65)
    print("Experiment A1: SCBM vs Random vs Sequential Mapping")
    print("="*65)

    MAPPINGS = {
        "SCBM (Proposed)": mapping_scbm,
        "Random Bijective": mapping_random,
        "Sequential Shift": mapping_sequential,
    }
    # Test on attacks where mapping matters most
    TEST_ATTACKS = ["cropping", "content_removal", "copy_move"]

    results = {mname: {atk: [] for atk in TEST_ATTACKS} for mname in MAPPINGS}
    tmp_dir = os.path.join(OUT_DIR, "tmp_scbm")
    os.makedirs(tmp_dir, exist_ok=True)

    for fpath in files:
        base = os.path.splitext(os.path.basename(fpath))[0]
        orig = cv2.imread(fpath)
        print(f"\n  [{base}]")

        for mname, mfn in MAPPINGS.items():
            wm = embed_img(orig, mfn)

            for atk in TEST_ATTACKS:
                attacked = do_attack(wm.copy(), atk)
                if attacked is None: continue
                rec = recover_img(attacked, mfn, branch_override=None)

                rp = psnr(orig, rec)
                rs = ssim_v(orig, rec)
                rm = ms_ssim(orig, rec)
                results[mname][atk].append((rp, rs, rm))
                print(f"    [{mname:<20}] {atk:<18} PSNR={rp:.2f} SSIM={rs:.4f}")

    # Write CSV
    csv_path = os.path.join(OUT_DIR, "Table_SCBM_Ablation.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Mapping Strategy","Attack",
                    "R-PSNR mean","R-PSNR std","R-PSNR CI+-",
                    "SSIM mean","SSIM std",
                    "MS-SSIM mean"])
        for mname in MAPPINGS:
            for atk in TEST_ATTACKS:
                vals = results[mname][atk]
                if not vals: continue
                ps=[v[0] for v in vals]; ss=[v[1] for v in vals]; ms=[v[2] for v in vals]
                pm,pstd,pc = ci95(ps)
                sm,sstd,_  = ci95(ss)
                mm,_,_     = ci95(ms)
                w.writerow([mname, ATTACK_LABELS[atk],
                            f"{pm:.2f}",f"{pstd:.2f}",f"{pc:.2f}",
                            f"{sm:.4f}",f"{sstd:.4f}",f"{mm:.4f}"])
    print(f"\n  Saved: {csv_path}")
    return results

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A2 — 3-WAY CLASSIFIER ABLATION
# ══════════════════════════════════════════════════════════════════════════════
def exp_A2_classifier(files):
    print("\n" + "="*65)
    print("Experiment A2: 3-Way Classifier Ablation")
    print("="*65)

    VARIANTS = {
        "No Classifier (always C)": "C",
        "2-Way: no Branch A":       "noA",
        "2-Way: no Branch B":       "noB",
        "Full DLSBM (3-Way)":       None,
    }

    results = {vname: {atk: [] for atk in ATTACKS} for vname in VARIANTS}
    tmp_dir = os.path.join(OUT_DIR, "tmp_scbm")  # reuse watermarked from A1

    for fpath in files:
        base = os.path.splitext(os.path.basename(fpath))[0]
        orig = cv2.imread(fpath)
        wm   = embed_img(orig, mapping_scbm)
        print(f"\n  [{base}]")

        atk_cache = {}
        for atk in ATTACKS:
            attacked = do_attack(wm.copy(), atk)
            if attacked is not None:
                atk_cache[atk] = attacked

        for vname, override in VARIANTS.items():
            for atk, attacked in atk_cache.items():
                rec = recover_img(attacked.copy(), mapping_scbm, branch_override=override)
                rp  = psnr(orig, rec)
                rs  = ssim_v(orig, rec)
                rm  = ms_ssim(orig, rec)
                results[vname][atk].append((rp, rs, rm))
            print(f"    [{vname:<28}] done")

    csv_path = os.path.join(OUT_DIR, "Table_Classifier_Ablation.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Classifier Variant","Attack",
                    "R-PSNR mean","R-PSNR std",
                    "SSIM mean","SSIM std",
                    "MS-SSIM mean"])
        for vname in VARIANTS:
            for atk in ATTACKS:
                vals = results[vname][atk]
                if not vals: continue
                ps=[v[0] for v in vals]; ss=[v[1] for v in vals]; ms=[v[2] for v in vals]
                pm,pstd,_ = ci95(ps)
                sm,sstd,_ = ci95(ss)
                mm,_,_    = ci95(ms)
                w.writerow([vname, ATTACK_LABELS[atk],
                            f"{pm:.2f}",f"{pstd:.2f}",
                            f"{sm:.4f}",f"{sstd:.4f}",
                            f"{mm:.4f}"])
    print(f"\n  Saved: {csv_path}")
    return results

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B — THRESHOLD SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════
def exp_B_threshold(files):
    print("\n" + "="*65)
    print("Experiment B: Threshold Sensitivity Analysis")
    print("="*65)

    TAU_THS  = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    ETA_THS  = [0.001, 0.003, 0.005, 0.010, 0.020, 0.050]

    # Step 1: Collect (tau, eta) for all images × attacks
    print("\n  Phase B1: Collecting (tau, eta) per image x attack ...")
    tau_eta_data = {atk: [] for atk in ATTACKS}  # list of (tau, eta) per image

    for fpath in files:
        base = os.path.splitext(os.path.basename(fpath))[0]
        orig = cv2.imread(fpath)
        wm   = embed_img(orig, mapping_scbm)

        for atk in ATTACKS:
            attacked = do_attack(wm.copy(), atk)
            if attacked is None: continue
            tau, eta, _ = classify(attacked, tau_th=0.85, eta_th=0.005)
            tau_eta_data[atk].append((tau, eta))
        print(f"    [{base}] done")

    # Step 2: For each (tau_th, eta_th) compute correct classification rate
    print("\n  Phase B2: Sweeping thresholds ...")
    sweep_rows = []
    for tau_th in TAU_THS:
        for eta_th in ETA_THS:
            correct = 0
            total   = 0
            per_atk = {atk: {"correct":0,"total":0} for atk in ATTACKS}
            for atk in ATTACKS:
                opt = OPTIMAL_BRANCH[atk]
                for (tau, eta) in tau_eta_data[atk]:
                    is_noise = eta > eta_th
                    is_jpeg  = tau > tau_th and not is_noise
                    pred = "A" if is_jpeg else ("B" if is_noise else "C")
                    match = int(pred == opt)
                    correct += match
                    total   += 1
                    per_atk[atk]["correct"] += match
                    per_atk[atk]["total"]   += 1
            acc = 100.0 * correct / total
            per_acc = {atk: 100.0*per_atk[atk]["correct"]/max(1,per_atk[atk]["total"])
                       for atk in ATTACKS}
            sweep_rows.append((tau_th, eta_th, acc, per_acc))
            print(f"    tau_th={tau_th*100:.0f}% eta_th={eta_th*100:.1f}%  "
                  f"overall_acc={acc:.1f}%")

    # Write heatmap CSV
    csv_path = os.path.join(OUT_DIR, "Table_Threshold_Sweep.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["tau_th (%)","eta_th (%)","Overall Acc (%)",
                    "ContentRm","CopyMove","Splicing","JPEG","Noise","Cropping"])
        for (tau_th, eta_th, acc, pa) in sweep_rows:
            w.writerow([f"{tau_th*100:.0f}", f"{eta_th*100:.1f}",
                        f"{acc:.1f}",
                        f"{pa['content_removal']:.1f}",
                        f"{pa['copy_move']:.1f}",
                        f"{pa['splicing']:.1f}",
                        f"{pa['jpeg']:.1f}",
                        f"{pa['noise']:.1f}",
                        f"{pa['cropping']:.1f}"])

    print(f"\n  Saved: {csv_path}")
    return sweep_rows, tau_eta_data

# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(a1_results, a2_results, sweep_rows):
    print("\n" + "="*65)
    print("SUMMARY — SCBM ABLATION (Cropping 40%)")
    print("="*65)
    print(f"{'Mapping':<25} {'R-PSNR':>8} {'SSIM':>7}")
    print("-"*42)
    for mname, mres in a1_results.items():
        vals = mres["cropping"]
        if vals:
            pm,_,_ = ci95([v[0] for v in vals])
            sm,_,_ = ci95([v[1] for v in vals])
            print(f"  {mname:<23} {pm:>7.2f} {sm:>7.4f}")

    print("\n" + "="*65)
    print("SUMMARY — CLASSIFIER ABLATION (mean over all attacks)")
    print("="*65)
    print(f"{'Variant':<30} {'Content Rm':>11} {'JPEG':>8} {'Noise':>8} {'Overall':>8}")
    print("-"*65)
    for vname, vres in a2_results.items():
        cr_p = np.mean([v[0] for v in vres["content_removal"]]) if vres["content_removal"] else 0
        jp_p = np.mean([v[0] for v in vres["jpeg"]]) if vres["jpeg"] else 0
        ns_p = np.mean([v[0] for v in vres["noise"]]) if vres["noise"] else 0
        all_p = np.mean([v[0] for atk in ATTACKS for v in vres[atk]])
        print(f"  {vname:<28} {cr_p:>10.2f} {jp_p:>8.2f} {ns_p:>8.2f} {all_p:>8.2f}")

    print("\n" + "="*65)
    print("SUMMARY — THRESHOLD SENSITIVITY (optimal region)")
    print("="*65)
    best = max(sweep_rows, key=lambda x: x[2])
    print(f"  Best accuracy: {best[2]:.1f}%  at  "
          f"tau_th={best[0]*100:.0f}%  eta_th={best[1]*100:.1f}%")
    # Show accuracy at proposed thresholds
    proposed = [r for r in sweep_rows if abs(r[0]-0.85)<0.01 and abs(r[1]-0.005)<0.001]
    if proposed:
        p = proposed[0]
        print(f"  Proposed thresholds (85%, 0.5%): accuracy = {p[2]:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = get_files()
    if not files:
        print(f"ERROR: no images in '{INPUT_DIR}/'"); return
    print(f"Images: {len(files)}")

    a1_results             = exp_A1_scbm(files)
    a2_results             = exp_A2_classifier(files)
    sweep_rows, te_data    = exp_B_threshold(files)

    print_summary(a1_results, a2_results, sweep_rows)
    print(f"\nAll outputs in '{OUT_DIR}/'")

if __name__ == "__main__":
    main()
