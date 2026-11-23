"""Draws docs/figures/overview.svg (README top figure):  python docs/figures/make_overview.py docs/figures/overview.svg
Two mixtures -> frozen network per molecule -> weighted merge -> points -> rating."""
import math, sys
INK, MUTED, GRID, BLUE, ORANGE, RED = "#111", "#555", "#ddd", "#2a78d6", "#eb6834", "#c0392b"
W, H = 980, 362
o = []
T = lambda x, y, s, size=13, anchor="middle", fill=INK, weight=None, style=None: o.append(
    f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{fill}"'
    + (f' font-weight="{weight}"' if weight else '') + (f' font-style="{style}"' if style else '') + f'>{s}</text>')
arrow = lambda x1, y1, x2, y2, c=INK, w=1.3, dash=None: o.append(
    f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="{w}"'
    + (f' stroke-dasharray="{dash}"' if dash else '') + f' marker-end="url(#ar{"b" if c == BLUE else "o" if c == ORANGE else "k"})"/>')


def hexmol(cx, cy, c, tail):
    pts = " ".join(f"{cx + 10 * math.cos(math.pi / 6 + k * math.pi / 3):.1f},{cy + 10 * math.sin(math.pi / 6 + k * math.pi / 3):.1f}" for k in range(6))
    o.append(f'<polygon points="{pts}" fill="#fff" stroke="{c}" stroke-width="1.6"/>')
    x0, y0 = cx + 10 * math.cos(-math.pi / 6), cy + 10 * math.sin(-math.pi / 6)
    o.append(f'<polyline points="{x0:.1f},{y0:.1f} {x0 + 8:.1f},{y0 - 5:.1f}' + (f' {x0 + 16:.1f},{y0:.1f}' if tail else '') + f'" fill="none" stroke="{c}" stroke-width="1.6"/>')


def strip(x, y, c, shades):
    for k, a in enumerate(shades):
        o.append(f'<rect x="{x + 7 * k:.1f}" y="{y - 5:.1f}" width="7" height="10" fill="{c}" fill-opacity="{a}" stroke="#fff" stroke-width="0.8"/>')


o.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W * 1.2:.0f}" height="{H * 1.2:.0f}" font-family="STIX Two Text, Times New Roman, serif" role="img" aria-label="Pony at a glance: each molecule of two mixtures passes through the frozen network to a hidden vector; molecules are weighted, merged into one point per mixture and scaled by a learned length; the rating is a line in the distance between the two points.">')
o.append(f'<rect width="{W}" height="{H}" fill="#fff"/>')
o.append('<defs>' + "".join(f'<marker id="ar{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="{c}"/></marker>' for k, c in (("k", INK), ("b", BLUE), ("o", ORANGE))) + '</defs>')

# stage bands
for x0, x1, title, sub in ((150, 432, "Stage 1: frozen network", "one molecule at a time"),
                           (442, 760, "Stage 2: mix in the hidden layer", "weights from atoms, sum, length")):
    o.append(f'<rect x="{x0}" y="8" width="{x1 - x0}" height="{H - 16}" rx="8" fill="#f7f7f5" stroke="#ccc" stroke-dasharray="4 3"/>')
    T((x0 + x1) / 2, 28, title, 14, weight=600); T((x0 + x1) / 2, 45, sub, 12, fill=MUTED)

mols = {"A": (BLUE, [102, 142, 182], [0.45, 0.3, 0.25]), "B": (ORANGE, [262, 302], [0.6, 0.4])}
shades = {("A", 0): [.9, .3, .6, .2, .8, .4], ("A", 1): [.3, .8, .2, .6, .3, .7], ("A", 2): [.6, .4, .9, .3, .2, .5],
          ("B", 0): [.2, .7, .4, .9, .5, .3], ("B", 1): [.8, .2, .3, .5, .9, .6]}
# mixtures
for name, (c, ys, _) in mols.items():
    o.append(f'<rect x="14" y="{ys[0] - 22}" width="112" height="{ys[-1] - ys[0] + 44}" rx="8" fill="#fff" stroke="{c}" stroke-width="1.4"/>')
    T(70, ys[0] - 30, f"Mixture {name}", 13.5, fill=c, weight=600)
    for k, y in enumerate(ys):
        hexmol(52, y, c, k % 2 == 0); T(98, y + 4, f"m{k + 1}", 12, fill=MUTED, style="italic")
# network box
o.append(f'<rect x="200" y="78" width="110" height="246" rx="10" fill="#fff" stroke="{INK}" stroke-width="1.3"/>')
for a, b in (((225, 150), (255, 120)), ((225, 150), (255, 195)), ((225, 240), (255, 195)), ((225, 240), (255, 270)),
             ((255, 120), (285, 160)), ((255, 195), (285, 160)), ((255, 195), (285, 230)), ((255, 270), (285, 230))):
    o.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="#bbb"/>')
for (x, y) in ((225, 150), (225, 240), (255, 120), (255, 195), (255, 270), (285, 160), (285, 230)):
    o.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#fff" stroke="{MUTED}"/>')
T(255, 96, "Pony network", 12.5, weight=600)
T(255, 338, "116,603 parameters, trained on", 11.5, fill=MUTED); T(255, 351, "odour words of single molecules", 11.5, fill=MUTED)
# molecule vectors
for name, (c, ys, sh) in mols.items():
    for k, y in enumerate(ys):
        arrow(128, y, 198, y, c, 1.1)
        o.append(f'<line x1="312" y1="{y}" x2="350" y2="{y}" stroke="{c}" stroke-width="1.1" marker-end="url(#ar{"b" if c == BLUE else "o"})"/>')
        strip(354, y, c, shades[(name, k)])
T(375, 74, "v(m): 64 units", 12, fill=MUTED, style="italic")
# stage 2: shares, sum, length
for name, (c, ys, sh) in mols.items():
    yc = sum(ys) / len(ys)
    for y, s in zip(ys, sh):
        o.append(f'<rect x="460" y="{y - 5}" width="{70 * s:.1f}" height="10" rx="2" fill="{c}" fill-opacity=".55"/>')
        T(460 + 70 * s + 5, y + 4, f"{s:.2f}", 11, "start", MUTED)
        o.append(f'<line x1="545" y1="{y}" x2="586" y2="{yc}" stroke="{c}" stroke-width="1.1"/>')
    o.append(f'<circle cx="596" cy="{yc}" r="11" fill="#fff" stroke="{c}" stroke-width="1.4"/>'); T(596, yc + 5, "Σ", 14, fill=c)
    arrow(608, yc, 640, yc, c, 1.3)
    o.append(f'<rect x="643" y="{yc - 13}" width="58" height="26" rx="6" fill="#fff" stroke="{c}" stroke-width="1.4"/>')
    T(672, yc + 5, f"× g({name})", 12.5, fill=c, style="italic")
T(500, 74, "share of each molecule", 12, fill=MUTED)
T(601, 338, "133 numbers per copy, fitted on", 11.5, fill=MUTED); T(601, 351, "rated mixture pairs", 11.5, fill=MUTED)
# hidden layer plot
o.append(f'<rect x="780" y="84" width="180" height="180" rx="10" fill="#fff" stroke="{INK}" stroke-width="1.3"/>')
T(870, 76, "hidden layer, 64 units", 12, fill=MUTED)
for x, y, c in ((812, 122, BLUE), (840, 168, BLUE), (806, 186, BLUE), (906, 236, ORANGE), (940, 176, ORANGE)):
    o.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{c}" fill-opacity=".35"/>')
pa, pb = (826, 150), (918, 200)
o.append(f'<line x1="{pa[0]}" y1="{pa[1]}" x2="{pb[0]}" y2="{pb[1]}" stroke="{INK}" stroke-width="1.4" stroke-dasharray="5 3"/>')
o.append(f'<circle cx="{pa[0]}" cy="{pa[1]}" r="7" fill="{BLUE}" stroke="#fff" stroke-width="2"/>')
o.append(f'<circle cx="{pb[0]}" cy="{pb[1]}" r="7" fill="{ORANGE}" stroke="#fff" stroke-width="2"/>')
T(pa[0] - 2, pa[1] - 13, "A", 13, fill=BLUE, weight=600); T(pb[0] + 2, pb[1] + 24, "B", 13, fill=ORANGE, weight=600)
T(884, 168, "d(A, B)", 12.5, style="italic")
ya, yb = sum(mols["A"][1]) / 3, sum(mols["B"][1]) / 2
o.append(f'<path d="M703 {ya} C 745 {ya}, 760 {pa[1]}, {pa[0] - 9} {pa[1]}" fill="none" stroke="{BLUE}" stroke-width="1.3" marker-end="url(#arb)"/>')
o.append(f'<path d="M703 {yb} C 760 {yb}, 860 {yb + 20}, {pb[0] - 3} {pb[1] + 9}" fill="none" stroke="{ORANGE}" stroke-width="1.3" marker-end="url(#aro)"/>')
# rating
o.append(f'<rect x="790" y="292" width="160" height="40" rx="8" fill="#fff" stroke="{INK}" stroke-width="1.5"/>')
T(870, 317, "rating = a · d(A, B) + c", 13.5, weight=600)
arrow(870, 266, 870, 289)
o.append('</svg>')
open(sys.argv[1], "w").write("".join(o))
