"""Tutorial protocol scheme (SVG). Usage:  python make_scheme.py tutorial-scheme.svg && inkscape -z tutorial-scheme.svg --export-png=tutorial-scheme.png --export-width=2200
, from the hand-drawn sketch: pools -> selection -> training set ->
finetuned teacher + UQ artifact -> distillation pool -> GRACE/FS student + active set -> LAMMPS."""
from pathlib import Path
import sys

W, H = 1620, 600
FONT = "Liberation Sans, Helvetica, Arial, sans-serif"
C = {  # fill, stroke
    "model": ("#dbe7f6", "#2f5f9e"),
    "data":  ("#fdf0d5", "#c8891a"),
    "uq":    ("#e9e0f5", "#6d4aa8"),
    "sim":   ("#dff1e3", "#2e8b57"),
}
INK, ARROW = "#1f2933", "#4a4a4a"
out = []
def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def box(x, y, w, h, title, lines, kind, tag=None):
    fill, stroke = C[kind]
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" ry="10" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    n = 1 + len(lines); lh = 17
    y0 = y + h/2 - (lh*(n-1) + 4)/2 + 6
    out.append(f'<text x="{x+w/2}" y="{y0}" text-anchor="middle" font-family="{FONT}" font-size="15.5" font-weight="bold" fill="{INK}">{esc(title)}</text>')
    for i, ln in enumerate(lines, 1):
        out.append(f'<text x="{x+w/2}" y="{y0 + 4 + i*lh}" text-anchor="middle" font-family="{FONT}" font-size="12.5" fill="{INK}">{esc(ln)}</text>')
    if tag:
        tw = 8 + 7.5*len(tag)
        out.append(f'<rect x="{x+w-tw-6}" y="{y-9}" width="{tw}" height="18" rx="9" fill="{stroke}"/>')
        out.append(f'<text x="{x+w-tw/2-6}" y="{y+4}" text-anchor="middle" font-family="{FONT}" font-size="11" font-weight="bold" fill="#fff">{esc(tag)}</text>')

def arrow(points, dashed=False, head=True):
    d = "M" + " L".join(f"{x},{y}" for x, y in points)
    extra = ' stroke-dasharray="6,5"' if dashed else ""
    m = ' marker-end="url(#arr)"' if head else ""
    out.append(f'<path d="{d}" fill="none" stroke="{ARROW}" stroke-width="1.8"{extra}{m}/>')

def label(x, y, text, anchor="middle", size=12, italic=True, color="#3a3a3a"):
    st = ' font-style="italic"' if italic else ""
    out.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{FONT}" font-size="{size}"{st} fill="{color}">{esc(text)}</text>')

def dot(x, y): out.append(f'<circle cx="{x}" cy="{y}" r="3.2" fill="{ARROW}"/>')

# ---------------------------------------------------------------- geometry
TOP, MID, BOT, BH = 60, 250, 400, 96
# column 1: foundation model above the two DFT pools
box(30, TOP, 220, BH, "GRACE-3L-OMAT-large", ["foundation model", "trained on 100+ M DFT structures"], "model", "§2")
out.append(f'<rect x="18" y="{MID-38}" width="244" height="{BOT+BH-MID+52}" rx="12" fill="none" stroke="#c8891a" stroke-width="1.2" stroke-dasharray="5,4"/>')
label(140, MID-20, "Al–Li DFT database (Menon et al. 2024)", size=12, italic=False, color="#8a5c0e")
box(30, MID, 220, BH, "Convex-hull pool", ["structures on and near", "the DFT convex hull"], "data", "§1")
box(30, BOT, 220, BH, "Deformation pool", ["1000 random strained and", "rattled structures"], "data", "§1")
# column 2: selection
box(300, MID, 180, BH, "128 structures", ["112 train + 16 test"], "data")
box(300, BOT, 180, BH, "FPS → 128 structures", ["farthest-point sampling in", "the foundation model's", "feature space"], "uq", "§3")
# column 3: training set (tall)
box(530, MID, 120, BOT+BH-MID, "Training set", ["256 DFT", "structures", "", "energies, forces,", "stresses"], "data")
# column 4: teacher and its UQ artifact
box(700, TOP, 180, BH, "GRACE-3L finetuned", ["the teacher", "hull + FPS data; only the", "readout weights train"], "model", "§4")
box(700, MID, 180, BH, "UQ artifact", ["Gaussian mixture on the", "teacher's features → γ"], "uq", "§5")
# column 5: distillation pool
box(980, TOP, 160, BH, "Distillation pool", ["deformed hull structures", "labelled by the teacher,", "kept if γ ≤ 5"], "data", "§9")
# column 6: student and its active set
box(1190, TOP, 140, BH, "GRACE/FS", ["the student", "fast linear model"], "model", "§10")
box(1190, MID, 140, BH, "Active set (ASI)", ["D-optimality → γ", "at every MD step"], "uq", "§10")
# column 7: LAMMPS
box(1430, TOP, 160, BH, "LAMMPS MD", ["CPU or GPU (Kokkos),", "extrapolation grade", "on the fly"], "sim", "§12")

# ---------------------------------------------------------------- arrows
y1 = TOP + BH/2   # top row centre line
# foundation model -> teacher, training set joins from below
arrow([(250, y1), (590, y1)], head=False); dot(590, y1)
arrow([(590, y1), (700, y1)])
arrow([(590, MID), (590, y1)], head=False)
label(470, y1-10, "finetuning", size=13)
# pools -> selection -> training set
ym, yb = MID + BH/2, BOT + BH/2
arrow([(250, ym), (300, ym)]); arrow([(250, yb), (300, yb)])
arrow([(480, ym), (530, ym)]); arrow([(480, yb), (530, yb)])
# training set -> UQ artifact (features of the training atoms)
arrow([(650, ym), (700, ym)])
# teacher -> UQ artifact (features); the UQ artifact joins the teacher's labels
arrow([(790, TOP+BH), (790, MID)]); label(798, (TOP+BH+MID)/2+4, "features", anchor="start")
arrow([(880, ym), (910, ym), (910, y1)], head=False); dot(910, y1)
label(918, ym-8, "γ", anchor="start", size=13)
# teacher (+UQ) -> distillation pool -> student
arrow([(880, y1), (980, y1)]); label(945, y1-10, "E, F, γ", size=12)
arrow([(1140, y1), (1190, y1)]); label(1165, y1-10, "fit", size=12)
# student -> active set; both -> LAMMPS
arrow([(1260, TOP+BH), (1260, MID)]); label(1268, (TOP+BH+MID)/2+4, "pace_activeset", anchor="start", size=11.5, italic=False)
arrow([(1330, ym), (1360, ym), (1360, y1)], head=False); dot(1360, y1)
arrow([(1330, y1), (1360, y1)], head=False)
arrow([(1360, y1), (1430, y1)]); label(1395, y1-10, "E, F, γ", size=12)
label(1368, ym-8, "γ", anchor="start", size=13)
# closing the loop: LAMMPS -> structures for the next DFT round -> pools
arrow([(1510, TOP+BH), (1510, 548), (140, 548), (140, BOT+BH)], dashed=True)
label(825, 541, "structures the student is unsure about → next DFT round (§12)", size=12.5)

# ---------------------------------------------------------------- legend
lx, ly = 705, 440
for i, (kind, name) in enumerate([("model", "models"), ("data", "DFT data"), ("uq", "uncertainty / selection"), ("sim", "simulation")]):
    fill, stroke = C[kind]
    x = lx + [0, 100, 210, 385][i]
    out.append(f'<rect x="{x}" y="{ly}" width="16" height="16" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    label(x + 22, ly + 12.5, name, anchor="start", size=12, italic=False)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
       f'<defs><marker id="arr" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="userSpaceOnUse">'
       f'<path d="M0,0 L9,4.5 L0,9 z" fill="{ARROW}"/></marker></defs>\n'
       f'<rect width="{W}" height="{H}" fill="#ffffff"/>\n' + "\n".join(out) + "\n</svg>\n")
dst = Path(sys.argv[1]); dst.write_text(svg); print("wrote", dst, len(svg), "bytes")
