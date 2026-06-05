"""
generate_system_overview_figure.py
===================================
Generates the combined DLSBM system-overview figure (new Fig. 0 / Fig. 1
for the manuscript revision) showing BOTH the embedding phase and the
detection & recovery phase in a single diagram.

Existing Figs 1-3 show the sub-process detail (embedding block diagram,
SCBM flowchart, recovery flowchart). This figure gives the high-level
end-to-end view that the reviewer is asking for.

Output: DLSBM_System_Overview.pdf  +  DLSBM_System_Overview.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patheffects as pe
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# ──────────────────────────────────────────────────────────────────────────
# Layout constants
# ──────────────────────────────────────────────────────────────────────────
FW, FH   = 20, 9.5      # figure width, height (inches)
DPI      = 300
EY       = 7.2          # embedding row  y-centre
DY       = 2.5          # detection row  y-centre
MID_Y    = (EY + DY) / 2  # midpoint between rows
BOX_H    = 1.40         # standard box height
BOX_W    = 2.10         # standard box width
BR_H     = 1.05         # branch box height
BR_W     = 2.40         # branch box width

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
FS       = 7.2                              # default font size

# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
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
    """Straight diagonal arrow."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=col, lw=lw), zorder=5)

def note(cx, cy, text, fs=6.0, col="#666666"):
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=col, style="italic", zorder=4)

# ──────────────────────────────────────────────────────────────────────────
# Phase background bands
# ──────────────────────────────────────────────────────────────────────────
ax.add_patch(Rectangle((0.15, EY - 1.75), FW - 0.30, 3.60,
             facecolor="#EBF5FB", alpha=0.50, zorder=0, lw=0))
ax.add_patch(Rectangle((0.15, 0.80), FW - 0.30, 3.63,
             facecolor="#E9F7EF", alpha=0.50, zorder=0, lw=0))

ax.text(FW / 2, EY + 1.82, "EMBEDDING PHASE", fontsize=9.5, fontweight="bold",
        color="#1A5276", va="center", ha="center")
ax.text(FW / 2, DY + 2.18, "DETECTION & RECOVERY PHASE", fontsize=9.5,
        fontweight="bold", color="#1E8449", va="center", ha="center")

ax.axhline(y=EY - 1.80, xmin=0.01, xmax=0.99,
           color="#BDC3C7", lw=0.8, ls="--", zorder=1)
ax.axhline(y=DY + 1.90, xmin=0.01, xmax=0.99,
           color="#BDC3C7", lw=0.8, ls="--", zorder=1)

# ──────────────────────────────────────────────────────────────────────────
# EMBEDDING PHASE  (top row, y = EY)
# ──────────────────────────────────────────────────────────────────────────
#   E1  E2   E3   E4     E5a/E5b   E6   E7   E8
# x: 1.3 3.65 6.0  8.35   11.1    13.8  16.3 18.8

E_X = [1.30, 3.65, 6.00, 8.35]           # linear boxes before branch
BRANCH_X = 11.10                          # x of branch boxes
E_X2  = [13.80, 16.30, 18.80]            # linear boxes after merge
BUY   = EY + 1.00                        # upper branch y
BLY   = EY - 1.00                        # lower branch y

# Secret-key dashed arrow into E4 (SCBM)
ax.text(8.35, EY + 2.00, "Secret Key (k)",
        ha="center", va="center", fontsize=6.0, color="#555555", style="italic")
ax.annotate("", xy=(8.35, EY + BOX_H / 2 + 0.05),
            xytext=(8.35, EY + 1.65),
            arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9,
                            linestyle="dashed"), zorder=5)

# ── E1: Original Image ──────────────────────────────────────────────────
box(E_X[0], EY, 1.90, BOX_H, "Original\nImage (I)", C_IMG, E_IMG)
harrow(E_X[0] + 0.95, EY, E_X[1] - BOX_W / 2 - 0.05)

# ── E2: Channel Separation & Block Division ──────────────────────────────
box(E_X[1], EY, BOX_W, BOX_H, "Channel Sep.\n& 4×4 Block\nDivision", C_PROC, E_PROC)
harrow(E_X[1] + BOX_W / 2, EY, E_X[2] - BOX_W / 2 - 0.05)

# ── E3: Clear 2 LSBs ─────────────────────────────────────────────────────
box(E_X[2], EY, BOX_W, BOX_H, "Clear 2 LSBs\n(Bᵢ & 0xFC)", C_PROC, E_PROC)
harrow(E_X[2] + BOX_W / 2, EY, E_X[3] - BOX_W / 2 - 0.05)

# ── E4: SCBM Partner Assignment ──────────────────────────────────────────
box(E_X[3], EY, BOX_W + 0.15, BOX_H, "SCBM Partner\nAssignment (Bⱼ)", C_PROC, E_PROC)

# fork arrows from E4 to branches
fork_x = E_X[3] + (BOX_W + 0.15) / 2
darrow(fork_x, EY + 0.12, BRANCH_X - BR_W / 2 - 0.05, BUY, col=EC)
darrow(fork_x, EY - 0.12, BRANCH_X - BR_W / 2 - 0.05, BLY, col=EC)

# ── E5a: 12-bit Auth Hash (upper branch) ─────────────────────────────────
box(BRANCH_X, BUY, BR_W, BR_H,
    "12-bit Auth Hash\nMD5(MSB₆(Bᵢ) ‖ ι)", C_HASH, E_HASH)

# ── E5b: 20-bit Recovery Payload (lower branch) ──────────────────────────
box(BRANCH_X, BLY, BR_W, BR_H,
    "20-bit Recovery\n2×2 sub-block\nmeans of Bⱼ", C_RECP, E_RECP, fs=6.8)

# merge arrows from branches to E6
darrow(BRANCH_X + BR_W / 2 + 0.05, BUY, E_X2[0] - BOX_W / 2 - 0.05, EY + 0.12, col=EC)
darrow(BRANCH_X + BR_W / 2 + 0.05, BLY, E_X2[0] - BOX_W / 2 - 0.05, EY - 0.12, col=EC)

# ── E6: 32-bit Payload Assembly ──────────────────────────────────────────
box(E_X2[0], EY, BOX_W + 0.20, BOX_H,
    "32-bit Payload\nAssembly\n(Auth ‖ Recovery)", C_HASH, E_HASH)
harrow(E_X2[0] + (BOX_W + 0.20) / 2, EY, E_X2[1] - BOX_W / 2 - 0.05)

# ── E7: 2-LSB Substitution ───────────────────────────────────────────────
box(E_X2[1], EY, BOX_W, BOX_H, "2-LSB\nSubstitution\ninto Bᵢ", C_PROC, E_PROC)
harrow(E_X2[1] + BOX_W / 2, EY, E_X2[2] - 1.05)

# ── E8: Watermarked Image ────────────────────────────────────────────────
box(E_X2[2], EY, 2.05, BOX_H, "Watermarked\nImage (Iᵂᵐ)", C_IMG, E_IMG)

# ──────────────────────────────────────────────────────────────────────────
# ATTACK CONNECTOR  (simple vertical: Watermarked → Attack → Tampered)
# D1 is placed directly below the Attack box; detection flows right → left.
# ──────────────────────────────────────────────────────────────────────────
ATK_X = 18.80
ATK_Y = MID_Y
ATK_W = 2.05
ATK_H = 1.15

# Arrow: bottom of Watermarked Image → top of Attack box
varrow(ATK_X, EY - BOX_H / 2 - 0.05, ATK_Y + ATK_H / 2 + 0.05,
       col=E_ATK, lw=1.5)

# Attack box
box(ATK_X, ATK_Y, ATK_W, ATK_H,
    "Adversarial\nAttack", C_ATK, E_ATK, fs=7.5, bold=True)

# Arrow: bottom of Attack box → top of Tampered Image (same x, no horizontal line)
varrow(ATK_X, ATK_Y - ATK_H / 2 - 0.05, DY + BOX_H / 2 + 0.05,
       col=E_ATK, lw=1.5)

# ──────────────────────────────────────────────────────────────────────────
# DETECTION & RECOVERY PHASE  (bottom row, RIGHT → LEFT)
# D1 (Tampered) is at x=18.80 directly below the Attack box.
# Steps 1→7 flow rightward-to-leftward; arrows point left (←).
# ──────────────────────────────────────────────────────────────────────────
D_X  = [18.80, 16.20, 13.60, 11.10]   # linear detection boxes, right → left
BR_X = 8.60                             # detection branch column x
D_X2 = [5.90, 3.40, 0.90]             # after branch merge: Merge, Inpaint, Output

BAY  = DY + 1.20    # Branch A  y (upper)
BBY  = DY           # Branch B  y (centre)
BCY  = DY - 1.20    # Branch C  y (lower)

# ── D1: Tampered Image ───────────────────────────────────────────────────
box(D_X[0], DY, 1.90, BOX_H, "Tampered\nImage (Iₐ)", C_ATK, E_ATK)
harrow(D_X[0] - 0.95, DY, D_X[1] + BOX_W / 2 + 0.05)   # arrow → LEFT

# ── D2: Block Extraction & Auth Check ────────────────────────────────────
box(D_X[1], DY, BOX_W, BOX_H, "Block Ext.\n& Auth. Check", C_PROC, E_PROC)
harrow(D_X[1] - BOX_W / 2, DY, D_X[2] + BOX_W / 2 + 0.05)

# ── D3: Binary Tamper Map ─────────────────────────────────────────────────
box(D_X[2], DY, BOX_W, BOX_H, "Binary\nTamper Map (T)\n(flag mismatches)", C_PROC, E_PROC)
harrow(D_X[2] - BOX_W / 2, DY, D_X[3] + (BOX_W + 0.20) / 2 + 0.05)

# ── D4: Attack Classification ─────────────────────────────────────────────
box(D_X[3], DY, BOX_W + 0.20, BOX_H, "Attack\nClassification\n(τ, η)", C_HASH, E_HASH)

# fork arrows from D4 LEFT side → branches (pointing left)
fork_dx = D_X[3] - (BOX_W + 0.20) / 2
darrow(fork_dx, DY + 0.14, BR_X + BR_W / 2 + 0.05, BAY, col=EC)
harrow(fork_dx, DY,         BR_X + BR_W / 2 + 0.05, col=EC)
darrow(fork_dx, DY - 0.14, BR_X + BR_W / 2 + 0.05, BCY, col=EC)

# ── Branch A: JPEG Bypass ────────────────────────────────────────────────
box(BR_X, BAY, BR_W, BR_H,
    "Branch A\nJPEG Bypass\n(τ > 85%,  η ≤ 0.5%)", C_BRA, E_BRA, fs=6.8)

# ── Branch B: Median Noise Repair ────────────────────────────────────────
box(BR_X, BBY, BR_W, BR_H,
    "Branch B\nNoise Repair\n(η > 0.5% — Median Filter)", C_BRB, E_BRB, fs=6.8)

# ── Branch C: SCBM Structural Recovery ───────────────────────────────────
box(BR_X, BCY, BR_W, BR_H,
    "Branch C\nSCBM Backup\n(Localized Structural)", C_BRC, E_BRC, fs=6.8)

# merge arrows from branches LEFT side → D5 (pointing left)
darrow(BR_X - BR_W / 2 - 0.05, BAY, D_X2[0] + BOX_W / 2 + 0.05, DY + 0.14, col=EC)
harrow(BR_X - BR_W / 2 - 0.05, DY,  D_X2[0] + BOX_W / 2 + 0.05, col=EC)
darrow(BR_X - BR_W / 2 - 0.05, BCY, D_X2[0] + BOX_W / 2 + 0.05, DY - 0.14, col=EC)

# ── D5: Merge Channels ───────────────────────────────────────────────────
box(D_X2[0], DY, BOX_W, BOX_H, "Merge\nChannels\n(R, G, B)", C_PROC, E_PROC)
harrow(D_X2[0] - BOX_W / 2, DY, D_X2[1] + BOX_W / 2 + 0.05)

# ── D6: Inpainting & Post-Process ────────────────────────────────────────
box(D_X2[1], DY, BOX_W, BOX_H, "Inpainting &\nPost-Process\n(Telea, r=2/5)", C_PROC, E_PROC)
harrow(D_X2[1] - BOX_W / 2, DY, D_X2[2] + 1.025 + 0.05)

# ── D7: Restored Image + Tamper Map (Output) ─────────────────────────────
box(D_X2[2], DY, 2.05, BOX_H,
    "Restored Image (Iᴿᵉᶜ)\n+ Tamper Map (T)", C_OUT, E_OUT)

# ──────────────────────────────────────────────────────────────────────────
# Step-number badges
# ──────────────────────────────────────────────────────────────────────────
def badge(cx, cy, n, color="#1A5276"):
    ax.add_patch(plt.Circle((cx, cy), 0.20, color=color, zorder=6))
    ax.text(cx, cy, str(n), ha="center", va="center",
            fontsize=6.5, color="white", fontweight="bold", zorder=7)

# Embedding step badges (top-left corner of each box)
badge(E_X[0] - 0.92 + 0.20, EY + BOX_H / 2 - 0.20, 1)
badge(E_X[1] - BOX_W / 2 + 0.20, EY + BOX_H / 2 - 0.20, 2)
badge(E_X[2] - BOX_W / 2 + 0.20, EY + BOX_H / 2 - 0.20, 3)
badge(E_X[3] - (BOX_W + 0.15) / 2 + 0.20, EY + BOX_H / 2 - 0.20, 4)
badge(BRANCH_X - BR_W / 2 + 0.20, BUY + BR_H / 2 - 0.18, 5)
badge(BRANCH_X - BR_W / 2 + 0.20, BLY + BR_H / 2 - 0.18, 5)
badge(E_X2[0] - (BOX_W + 0.20) / 2 + 0.20, EY + BOX_H / 2 - 0.20, 6)
badge(E_X2[1] - BOX_W / 2 + 0.20, EY + BOX_H / 2 - 0.20, 7)

# Detection step badges  (D1 at right; steps 1→7 follow the leftward arrow flow)
badge(D_X[0]  - 0.95 + 0.20,           DY + BOX_H / 2 - 0.20, 1, color="#1E8449")
badge(D_X[1]  - BOX_W / 2 + 0.20,      DY + BOX_H / 2 - 0.20, 2, color="#1E8449")
badge(D_X[2]  - BOX_W / 2 + 0.20,      DY + BOX_H / 2 - 0.20, 3, color="#1E8449")
badge(D_X[3]  - (BOX_W + 0.20)/2 + 0.20, DY + BOX_H / 2 - 0.20, 4, color="#1E8449")
badge(BR_X    - BR_W / 2 + 0.20,        BAY + BR_H / 2 - 0.18, 5, color="#1E8449")
badge(D_X2[0] - BOX_W / 2 + 0.20,      DY + BOX_H / 2 - 0.20, 6, color="#1E8449")
badge(D_X2[1] - BOX_W / 2 + 0.20,      DY + BOX_H / 2 - 0.20, 7, color="#1E8449")

# ──────────────────────────────────────────────────────────────────────────
# Legend
# ──────────────────────────────────────────────────────────────────────────
LGND_X = 0.55
LGND_Y = 0.52
items = [
    (C_IMG,  E_IMG,  "Input / Output Image"),
    (C_PROC, E_PROC, "Processing Step"),
    (C_HASH, E_HASH, "Auth. Hash / Payload"),
    (C_RECP, E_RECP, "Recovery Payload"),
    (C_ATK,  E_ATK,  "Attack"),
    (C_OUT,  E_OUT,  "System Output"),
]
# Explicit x-centres: Auth Hash (i=2) shifted left, Recovery Payload (i=3) shifted right
# to clear the Branch C detection box (x=7.40–9.80) that sits just above the legend.
LX_CENTRES = [0.55, 3.70, 6.10, 10.55, 13.15, 16.30]

for i, (fc, ec, label) in enumerate(items):
    lx = LX_CENTRES[i]
    p = FancyBboxPatch((lx - 0.40, LGND_Y - 0.22), 0.80, 0.44,
                        boxstyle="round,pad=0.05",
                        facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=3)
    ax.add_patch(p)
    ax.text(lx + 0.55, LGND_Y, label, ha="left", va="center",
            fontsize=6.0, color="#333333")

# ──────────────────────────────────────────────────────────────────────────
# Figure caption note (bottom)
# ──────────────────────────────────────────────────────────────────────────
ax.text(FW / 2, 0.07,
        "Fig. X. End-to-end overview of the DLSBM framework. "
        "Top (blue): Embedding Phase (steps 1–7). "
        "Bottom (green): Detection & Recovery Phase (steps 1–7). "
        "Sub-process details are given in Figs. 1–3 and Algorithms 1–2.",
        ha="center", va="bottom", fontsize=6.5, color="#444444",
        style="italic")

# ──────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────
out_base = "DLSBM_System_Overview"
plt.tight_layout(pad=0.4)
plt.savefig(out_base + ".pdf", dpi=DPI, bbox_inches="tight",
            facecolor="white")
plt.savefig(out_base + ".png", dpi=DPI, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print(f"Saved {out_base}.pdf and {out_base}.png")
