"""
merge_lake_grids.py
====================
Merges the geometric and compound attack result grids for the Lake image
into a single PDF with caption pages.

Structure of output PDF
-----------------------
  Page 1  : Caption for geometric transformations section
  Page 2  : Grid — 13 geometric attacks (Original | Attacked | Tamper Map | Recovered)
  Page 3  : Caption for compound attacks section
  Page 4  : Grid — 9 compound (mixed) attacks

Output
------
  geometric_compound_grids/Grid_Lake_MERGED.pdf

Run from:  C:/Users/tiwar/Downloads/journal implementation/
"""

import os, io
import textwrap
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

import pypdf

# ── Paths ─────────────────────────────────────────────────────────────────────
GEO_PDF = os.path.join("geometric_compound_grids", "geometric", "Grid_Lake.pdf")
CMP_PDF = os.path.join("geometric_compound_grids", "compound",  "Grid_Lake.pdf")
GEO_PNG = os.path.join("geometric_compound_grids", "geometric", "Grid_Lake.png")
CMP_PNG = os.path.join("geometric_compound_grids", "compound",  "Grid_Lake.png")
OUT_PDF = os.path.join("geometric_compound_grids", "Grid_Lake_MERGED.pdf")

# ── Caption text ──────────────────────────────────────────────────────────────
GEO_TITLE = (
    "Supplementary Figure S10 (Part 1 of 2): "
    "DLSBM Robustness Under Geometric Transformations — USC-SIPI Lake Image"
)

GEO_BODY = """\
This figure presents DLSBM tamper detection and recovery results for 13 geometric \
transformation attacks applied to the watermarked USC-SIPI Lake image \
(512 × 512, colour). Each row corresponds to one attack; results are shown across \
four columns described below.

Column Descriptions
───────────────────
(a) Original      Clean host image before watermark embedding.
(b) Attacked      The geometric transformation applied to the watermarked image. \
Rotations use BORDER_REFLECT padding to avoid black-border artefacts; \
scaling attacks downsample then upsample back to the original resolution.
(c) Tamper Map    Block-level detection output from DLSBM Pass-1 hash verification. \
Red blocks (4×4 pixels each) indicate regions flagged as tampered by the \
MD5 location-dependent hash check; grey areas pass authentication.
(d) Recovered     DLSBM restoration output. Per-image R-PSNR (dB) and SSIM are \
annotated in the bottom-left corner of each cell.

Key Findings
────────────
• All 13 geometric attacks achieve τ = 100 % tamper detection (solid-red tamper maps), \
confirming that DLSBM is fully sensitive to any geometric modification of pixel values \
or block alignments — the defining property of fragile watermarking.

• The 3-way classifier consistently routes all cases to Branch A (global-modification \
bypass) because τ > 85 % and η ≈ 0 % (no salt-and-pepper noise introduced). \
This is the correct classification: the system correctly identifies a uniform, \
global transformation rather than localised forgery.

• Rotation and flip attacks (rows 1–5, 10–11) yield R-PSNR of 11–16 dB because \
the SCBM backup blocks are co-located in the same spatial domain and are destroyed \
by the same transformation, precluding per-block recovery. This is an inherent \
trade-off of spatial-domain fragile watermarking (cf. robust watermarking).

• Resampling attacks (scaling ×0.50 to ×1.50, rows 6–9) achieve substantially \
better recovery (R-PSNR 29–37 dB, SSIM 0.87–0.97) because mild pixel-value changes \
from bilinear interpolation partially preserve block statistics, allowing Branch A \
to restore many blocks from the intact backup stream.

• Combined rotation + scaling attacks (rows 12–13) behave similarly to pure \
rotation, confirming that rotation dominates recovery difficulty.
"""

CMP_TITLE = (
    "Supplementary Figure S10 (Part 2 of 2): "
    "DLSBM 3-Way Classification Under Compound (Mixed) Attacks — USC-SIPI Lake Image"
)

CMP_BODY = """\
This figure presents DLSBM tamper detection and recovery results for 9 compound \
(mixed) attacks applied to the same watermarked Lake image. Each row combines two \
simultaneous attack components. The row label includes the 3-way branch selected \
by the DLSBM classifier [Br. A or Br. B]. Branch selection was unanimous \
(identical across all 9 USC-SIPI images) for every scenario.

Column Descriptions
───────────────────
(a) Original      Clean host image.
(b) Attacked      Compound attack: structural forgery followed by secondary JPEG \
compression or salt-and-pepper (S&P) noise. Black regions in some cells indicate \
zeroed-out crop or content-removal areas.
(c) Tamper Map    DLSBM block-level detection. Solid red = global pixel-level change \
(e.g. JPEG affecting all blocks). Scattered/speckled red patterns arise when S&P \
noise changes individual pixels in otherwise-intact blocks.
(d) Recovered     DLSBM output annotated with R-PSNR and SSIM.

Branch Classification Logic
───────────────────────────
  Branch A (JPEG Bypass)         τ > 85 % and η ≤ 0.5 %   → global transform, no noise
  Branch B (Noise Repair)        η > 0.5 %                 → salt-and-pepper noise present
  Branch C (Structural Recovery) τ ≤ 85 % and η ≤ 0.5 %   → localised forgery only

Key Findings
────────────
• All 9 scenarios yield unanimous branch consensus across 9 images, demonstrating \
deterministic classification robustness under compound attacks.

• Noise-bearing combinations (rows 1–2, 4, 6, 8–9) route to Branch B. The \
median-filter denoising pass cleans S&P residuals before per-block reconstruction. \
JPEG+S&P attacks recover at 35–38 dB (SSIM ≈ 0.96).

• Copy-Move + JPEG (row 7, [Br. A]) is the only case routed to Branch A because \
copy-move alone does not introduce extreme pixels (η = 0.03 %), and JPEG drives \
τ to 100 %. Branch A correctly treats this as a global-compression event and \
preserves the image with per-block restoration for zero-mean backup blocks.

• Crop-combined attacks (rows 3–4, R-PSNR ≈ 17 dB) suffer the most: the 30 % \
cropping zeroes out a large border region that contains many SCBM backup blocks, \
eliminating the recovery data needed by both Branch A and Branch B. This limitation \
is explicitly noted in Table S10.

• Branch C is absent from all compound scenarios because every compound attack \
includes a secondary global component (JPEG or noise) that pushes τ > 85 % or \
η > 0.5 %. Branch C remains active for single-component structural attacks \
(content removal, copy-move, splicing) as reported in Supplementary Tables S2–S5.
"""

# ── Helper: create one caption page as an in-memory PDF bytes ─────────────────
def caption_page_bytes(title, body):
    """Render title + body text as A4 portrait matplotlib figure → PDF bytes."""
    fig = plt.figure(figsize=(8.27, 11.69))   # A4 portrait in inches
    fig.patch.set_facecolor("white")

    # title block
    fig.text(
        0.08, 0.94, title,
        fontsize=11, fontweight="bold", color="#1a1a2e",
        wrap=True, va="top", ha="left",
        transform=fig.transFigure,
    )

    # thin horizontal rule under title
    fig.add_artist(
        plt.Line2D([0.08, 0.92], [0.92, 0.92],
                   transform=fig.transFigure,
                   color="#444444", linewidth=0.8)
    )

    # body text — wrap manually so it fits A4
    wrapped = []
    for para in body.split("\n"):
        if para.strip() == "":
            wrapped.append("")
        elif para.startswith(" ") or para.startswith("─") or para.startswith("  "):
            wrapped.append(para)          # preserve indented / rule lines
        else:
            for line in textwrap.wrap(para, width=95):
                wrapped.append(line)

    body_text = "\n".join(wrapped)
    fig.text(
        0.08, 0.90, body_text,
        fontsize=8.2, va="top", ha="left",
        transform=fig.transFigure,
        fontfamily="monospace",
        color="#222222",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Merge ─────────────────────────────────────────────────────────────────────
def main():
    print("Building caption pages ...")
    geo_cap_bytes = caption_page_bytes(GEO_TITLE, GEO_BODY)
    cmp_cap_bytes = caption_page_bytes(CMP_TITLE, CMP_BODY)

    writer = pypdf.PdfWriter()

    def add_pdf_bytes(b):
        reader = pypdf.PdfReader(io.BytesIO(b))
        for page in reader.pages:
            writer.add_page(page)

    def add_pdf_file(path):
        reader = pypdf.PdfReader(path)
        for page in reader.pages:
            writer.add_page(page)

    # Page 1: geometric caption
    print("Adding geometric caption (page 1) ...")
    add_pdf_bytes(geo_cap_bytes)

    # Page 2+: geometric grid (may be multi-page if matplotlib split it)
    print("Adding geometric grid ...")
    add_pdf_file(GEO_PDF)

    # Next page: compound caption
    print("Adding compound caption ...")
    add_pdf_bytes(cmp_cap_bytes)

    # Final pages: compound grid
    print("Adding compound grid ...")
    add_pdf_file(CMP_PDF)

    with open(OUT_PDF, "wb") as f:
        writer.write(f)

    size_mb = os.path.getsize(OUT_PDF) / (1024 * 1024)
    print(f"\nMerged PDF saved: {OUT_PDF}  ({size_mb:.1f} MB)")
    print(f"Total pages: {len(writer.pages)}")


if __name__ == "__main__":
    main()
