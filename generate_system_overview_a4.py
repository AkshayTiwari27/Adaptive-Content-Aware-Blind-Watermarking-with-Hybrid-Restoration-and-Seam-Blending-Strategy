import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import sys

# ──────────────────────────────────────────────────────────────────────────
# Layout constants
# ──────────────────────────────────────────────────────────────────────────
FW, FH   = 8.27, 11.69  # A4 portrait size
DPI      = 300
BOX_H    = 0.90
BOX_W    = 2.05
BR_H     = 0.65
BR_W     = 2.05

# ── Colour palette ──────────────────────────────────────────────────────
C_IMG    = "#D5E8D4"; E_IMG  = "#82B366"   # green  – image nodes
C_PROC   = "#DAE8FC"; E_PROC = "#6C8EBF"   # blue   – process steps
C_HASH   = "#FFE6CC"; E_HASH = "#D79B00"   # orange – hash/auth
C_RECP   = "#FFF2CC"; E_RECP = "#D6B656"   # yellow – recovery payload
C_ATK    = "#F8CECC"; E_ATK  = "#B85450"   # red    – attack
C_BRA    = "#DAE8FC"; E_BRA  = "#6C8EBF"   # blue   – Branch A
C_BRB    = "#D5E8D4"; E_BRB  = "#82B366"   # green  – Branch B
C_BRC    = "#FFE6CC"; E_BRC  = "#D6B656"   # yellow – Branch C
C_OUT    = "#E1D5E7"; E_OUT  = "#9673A6"   # purple – outputs
EC       = "#333333"                        # default edge
FS       = 7.5                              # default font size

fig, ax = plt.subplots(figsize=(FW, FH), dpi=DPI)
ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis("off")
fig.patch.set_facecolor("white")

def box(cx, cy, w, h, text, fc, ec=EC, fs=FS, bold=False):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                        boxstyle="round,pad=0.10",
                        facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=3)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#111111",
            multialignment="center", linespacing=1.25, zorder=4)

def harrow(x1, y, x2, col=EC, lw=1.25):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="->", color=col, lw=lw), zorder=5)

def varrow(x, y1, y2, col=EC, lw=1.25):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="->", color=col, lw=lw), zorder=5)

def darrow(x1, y1, x2, y2, col=EC, lw=1.25):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=col, lw=lw), zorder=5)

def badge(cx, cy, n, color="#1A5276"):
    ax.add_patch(plt.Circle((cx, cy), 0.18, color=color, zorder=6))
    ax.text(cx, cy, str(n), ha="center", va="center",
            fontsize=6.5, color="white", fontweight="bold", zorder=7)

# Columns
C1 = 1.45
C2 = 4.135
C3 = 6.82

# Rows
R1 = 10.65
R2 = 8.9
R3 = 7.2
R4 = 5.6
R5 = 3.7
R6 = 1.8

# Background bands
ax.add_patch(Rectangle((0.15, 6.6), FW - 0.30, 5.1,
             facecolor="#EBF5FB", alpha=0.50, zorder=0, lw=0))
ax.add_patch(Rectangle((0.15, 1.1), FW - 0.30, 5.5,
             facecolor="#E9F7EF", alpha=0.50, zorder=0, lw=0))

ax.text(FW / 2, 11.45, "EMBEDDING PHASE", fontsize=10, fontweight="bold",
        color="#1A5276", va="center", ha="center")
ax.text(FW / 2, 6.4, "DETECTION & RECOVERY PHASE", fontsize=10,
        fontweight="bold", color="#1E8449", va="center", ha="center")

# EMBEDDING PHASE
box(C1, R1, BOX_W-0.2, BOX_H, "Original\nImage (I)", C_IMG, E_IMG)
badge(C1 - (BOX_W-0.2)/2 + 0.15, R1 + BOX_H/2 - 0.15, 1)

box(C2, R1, BOX_W, BOX_H, "Channel Sep.\n& 4×4 Block\nDivision", C_PROC, E_PROC)
badge(C2 - BOX_W/2 + 0.15, R1 + BOX_H/2 - 0.15, 2)

box(C3, R1, BOX_W, BOX_H, "Clear 2 LSBs\n(Bᵢ & 0xFC)", C_PROC, E_PROC)
badge(C3 - BOX_W/2 + 0.15, R1 + BOX_H/2 - 0.15, 3)

harrow(C1 + (BOX_W-0.2)/2 + 0.05, R1, C2 - BOX_W/2 - 0.05)
harrow(C2 + BOX_W/2 + 0.05, R1, C3 - BOX_W/2 - 0.05)

varrow(C3, R1 - BOX_H/2 - 0.05, R2 + BOX_H/2 + 0.05)

box(C3, R2, BOX_W, BOX_H, "SCBM Partner\nAssignment (Bⱼ)", C_PROC, E_PROC)
badge(C3 - BOX_W/2 + 0.15, R2 + BOX_H/2 - 0.15, 4)

ax.text(C3+1.2, R2+0.8, "Secret Key (k)", ha="center", va="center", fontsize=6.5, color="#555555", style="italic")
ax.annotate("", xy=(C3, R2 + BOX_H/2 + 0.05), xytext=(C3+0.8, R2+0.8),
            arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9, linestyle="dashed"), zorder=5)

Y_E5a = R2 + 0.6
Y_E5b = R2 - 0.6
box(C2, Y_E5a, BR_W, BR_H, "12-bit Auth Hash\nMD5(MSB₆(Bᵢ) ‖ ι)", C_HASH, E_HASH)
box(C2, Y_E5b, BR_W, BR_H, "20-bit Recovery\n2×2 sub-block\nmeans of Bⱼ", C_RECP, E_RECP, fs=6.5)
badge(C2 - BR_W/2 + 0.15, Y_E5a + BR_H/2 - 0.15, 5)
badge(C2 - BR_W/2 + 0.15, Y_E5b + BR_H/2 - 0.15, 5)

darrow(C3 - BOX_W/2 - 0.05, R2 + 0.1, C2 + BR_W/2 + 0.05, Y_E5a)
darrow(C3 - BOX_W/2 - 0.05, R2 - 0.1, C2 + BR_W/2 + 0.05, Y_E5b)

box(C1, R2, BOX_W, BOX_H, "32-bit Payload\nAssembly\n(Auth ‖ Recovery)", C_HASH, E_HASH)
badge(C1 - BOX_W/2 + 0.15, R2 + BOX_H/2 - 0.15, 6)

darrow(C2 - BR_W/2 - 0.05, Y_E5a, C1 + BOX_W/2 + 0.05, R2 + 0.1)
darrow(C2 - BR_W/2 - 0.05, Y_E5b, C1 + BOX_W/2 + 0.05, R2 - 0.1)

varrow(C1, R2 - BOX_H/2 - 0.05, R3 + BOX_H/2 + 0.05)

box(C1, R3, BOX_W, BOX_H, "2-LSB\nSubstitution\ninto Bᵢ", C_PROC, E_PROC)
badge(C1 - BOX_W/2 + 0.15, R3 + BOX_H/2 - 0.15, 7)

box(C2, R3, BOX_W-0.2, BOX_H, "Watermarked\nImage (Iᵂᵐ)", C_IMG, E_IMG)
box(C3, R3, BOX_W-0.2, BOX_H, "Adversarial\nAttack", C_ATK, E_ATK, fs=8.0, bold=True)

harrow(C1 + BOX_W/2 + 0.05, R3, C2 - (BOX_W-0.2)/2 - 0.05)
harrow(C2 + (BOX_W-0.2)/2 + 0.05, R3, C3 - (BOX_W-0.2)/2 - 0.05, col=E_ATK, lw=1.5)

varrow(C3, R3 - BOX_H/2 - 0.05, R4 + BOX_H/2 + 0.05, col=E_ATK, lw=1.5)


# DETECTION PHASE
box(C3, R4, BOX_W-0.2, BOX_H, "Tampered\nImage (Iₐ)", C_ATK, E_ATK)
badge(C3 - (BOX_W-0.2)/2 + 0.15, R4 + BOX_H/2 - 0.15, 1, color="#1E8449")

box(C2, R4, BOX_W, BOX_H, "Block Ext.\n& Auth. Check", C_PROC, E_PROC)
badge(C2 - BOX_W/2 + 0.15, R4 + BOX_H/2 - 0.15, 2, color="#1E8449")

box(C1, R4, BOX_W, BOX_H, "Binary\nTamper Map (T)\n(flag mismatches)", C_PROC, E_PROC)
badge(C1 - BOX_W/2 + 0.15, R4 + BOX_H/2 - 0.15, 3, color="#1E8449")

harrow(C3 - (BOX_W-0.2)/2 - 0.05, R4, C2 + BOX_W/2 + 0.05)
harrow(C2 - BOX_W/2 - 0.05, R4, C1 + BOX_W/2 + 0.05)

varrow(C1, R4 - BOX_H/2 - 0.05, R5 + BOX_H/2 + 0.05)

box(C1, R5, BOX_W, BOX_H, "Attack\nClassification\n(τ, η)", C_HASH, E_HASH)
badge(C1 - BOX_W/2 + 0.15, R5 + BOX_H/2 - 0.15, 4, color="#1E8449")

Y_BrA = R5 + 0.75
Y_BrB = R5
Y_BrC = R5 - 0.75

box(C2, Y_BrA, BR_W, BR_H, "Branch A\nJPEG Bypass\n(τ > 85%,  η ≤ 0.5%)", C_BRA, E_BRA, fs=6.5)
box(C2, Y_BrB, BR_W, BR_H, "Branch B\nNoise Repair\n(η > 0.5% — Median Filter)", C_BRB, E_BRB, fs=6.5)
box(C2, Y_BrC, BR_W, BR_H, "Branch C\nSCBM Backup\n(Localized Structural)", C_BRC, E_BRC, fs=6.5)

badge(C2 - BR_W/2 + 0.15, Y_BrA + BR_H/2 - 0.15, 5, color="#1E8449")

darrow(C1 + BOX_W/2 + 0.05, R5 + 0.1, C2 - BR_W/2 - 0.05, Y_BrA)
harrow(C1 + BOX_W/2 + 0.05, R5,       C2 - BR_W/2 - 0.05)
darrow(C1 + BOX_W/2 + 0.05, R5 - 0.1, C2 - BR_W/2 - 0.05, Y_BrC)

box(C3, R5, BOX_W, BOX_H, "Merge\nChannels\n(R, G, B)", C_PROC, E_PROC)
badge(C3 - BOX_W/2 + 0.15, R5 + BOX_H/2 - 0.15, 6, color="#1E8449")

darrow(C2 + BR_W/2 + 0.05, Y_BrA, C3 - BOX_W/2 - 0.05, R5 + 0.1)
harrow(C2 + BR_W/2 + 0.05, R5,    C3 - BOX_W/2 - 0.05)
darrow(C2 + BR_W/2 + 0.05, Y_BrC, C3 - BOX_W/2 - 0.05, R5 - 0.1)

varrow(C3, R5 - BOX_H/2 - 0.05, R6 + BOX_H/2 + 0.05)

box(C3, R6, BOX_W, BOX_H, "Inpainting &\nPost-Process\n(Telea, r=2/5)", C_PROC, E_PROC)
badge(C3 - BOX_W/2 + 0.15, R6 + BOX_H/2 - 0.15, 7, color="#1E8449")

C12_MID = (C1 + C2) / 2
box(C12_MID, R6, C2 - C1 + BOX_W, BOX_H, "Restored Image (Iᴿᵉᶜ) + Tamper Map (T)", C_OUT, E_OUT, fs=8.5, bold=True)

harrow(C3 - BOX_W/2 - 0.05, R6, C12_MID + (C2 - C1 + BOX_W)/2 + 0.05)

# Legend
LGND_Y = 0.75
items = [
    (C_IMG,  E_IMG,  "Image/Output"),
    (C_PROC, E_PROC, "Process"),
    (C_HASH, E_HASH, "Auth. Hash"),
    (C_RECP, E_RECP, "Recovery"),
    (C_ATK,  E_ATK,  "Attack"),
    (C_OUT,  E_OUT,  "Final Result"),
]

for i, (fc, ec, label) in enumerate(items):
    lx = 0.5 + i * 1.2
    p = FancyBboxPatch((lx, LGND_Y - 0.15), 0.30, 0.30,
                        boxstyle="round,pad=0.05",
                        facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=3)
    ax.add_patch(p)
    ax.text(lx + 0.40, LGND_Y, label, ha="left", va="center",
            fontsize=6.5, color="#333333")

# Caption removed

out_base = "DLSBM_System_Overview_A4"
plt.tight_layout(pad=0.4)
plt.savefig(out_base + ".pdf", dpi=DPI, bbox_inches="tight", facecolor="white")
plt.savefig(out_base + ".png", dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved {out_base}.pdf and {out_base}.png")
