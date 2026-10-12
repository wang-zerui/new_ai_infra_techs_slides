"""Generate fig_k3_stack.svg: Kimi K3 attention-layer layout (69 KDA + 24 Gated MLA), slide scale (1 unit = 1pt).
Layer pattern from K3 report Table 1 and Sec. 2.1: 23 blocks of [3 KDA + 1 Gated MLA] + 1 final Gated MLA."""
import pathlib
ACC, ACC_F, GRAY, GRAY_F, INK = "#1F4E79", "#DEEBF7", "#595959", "#F2F2F2", "#222222"
W, H = 870, 285
o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H}pt" viewBox="0 0 {W} {H}" '
     'font-family="Helvetica, Arial, sans-serif">']
# layer bars
x0, step, bw, y, h = 40, 8.5, 7, 34, 46
layers = (["KDA"] * 3 + ["MLA"]) * 23 + ["MLA"]
assert len(layers) == 93 and layers.count("KDA") == 69
for i, kind in enumerate(layers):
    x = x0 + i * step
    fill = ACC if kind == "KDA" else "#BFBFBF"
    o.append(f'<rect x="{x:.1f}" y="{y}" width="{bw}" height="{h}" fill="{fill}" stroke="none"/>')
last_x = x0 + 92 * step + bw
o.append(f'<text x="{x0}" y="{y-10}" font-size="18" fill="{INK}">Layer 1</text>')
o.append(f'<text x="{last_x:.1f}" y="{y-10}" font-size="18" fill="{INK}" text-anchor="end">Layer 93</text>')
# bracket around block 1
bx1, bx2 = x0 - 3, x0 + 3 * step + bw + 3
o.append(f'<rect x="{bx1}" y="{y-4}" width="{bx2-bx1:.1f}" height="{h+8}" fill="none" stroke="{INK}" stroke-width="1.5" stroke-dasharray="4,3"/>')
# zoom container
zx, zy, zw, zh = 40, 120, 560, 158
o.append(f'<line x1="{bx1}" y1="{y+h+4}" x2="{zx}" y2="{zy}" stroke="{GRAY}" stroke-width="1.2" stroke-dasharray="4,3"/>')
o.append(f'<line x1="{bx2:.1f}" y1="{y+h+4}" x2="{zx+zw}" y2="{zy}" stroke="{GRAY}" stroke-width="1.2" stroke-dasharray="4,3"/>')
o.append(f'<rect x="{zx}" y="{zy}" width="{zw}" height="{zh}" rx="6" fill="#FFFFFF" stroke="{GRAY}" stroke-width="1.5" stroke-dasharray="6,4"/>')
cw, gap = 124, 12
cx0 = zx + (zw - (4 * cw + 3 * gap)) / 2
for j in range(4):
    cx = cx0 + j * (cw + gap)
    kda = j < 3
    f, s = (ACC_F, ACC) if kda else (GRAY_F, GRAY)
    o.append(f'<rect x="{cx:.1f}" y="{zy+14}" width="{cw}" height="58" rx="5" fill="{f}" stroke="{s}" stroke-width="2"/>')
    if kda:
        o.append(f'<text x="{cx+cw/2:.1f}" y="{zy+14+29+7:.1f}" font-size="20" font-weight="bold" fill="{ACC}" text-anchor="middle">KDA</text>')
    else:
        o.append(f'<text x="{cx+cw/2:.1f}" y="{zy+14+24:.1f}" font-size="19" font-weight="bold" fill="{INK}" text-anchor="middle">Gated MLA</text>')
        o.append(f'<text x="{cx+cw/2:.1f}" y="{zy+14+47:.1f}" font-size="17" fill="{INK}" text-anchor="middle">(NoPE)</text>')
    o.append(f'<line x1="{cx+cw/2:.1f}" y1="{zy+74}" x2="{cx+cw/2:.1f}" y2="{zy+88}" stroke="{INK}" stroke-width="2" marker-end="url(#a)"/>')
    o.append(f'<rect x="{cx:.1f}" y="{zy+90}" width="{cw}" height="52" rx="5" fill="#FFFFFF" stroke="{GRAY}" stroke-width="1.5"/>')
    o.append(f'<text x="{cx+cw/2:.1f}" y="{zy+90+32:.1f}" font-size="18" fill="{INK}" text-anchor="middle">LatentMoE</text>')
# legend
lx, ly = 640, 150
o.append(f'<rect x="{lx}" y="{ly}" width="16" height="34" fill="{ACC}"/>')
o.append(f'<text x="{lx+28}" y="{ly+15}" font-size="20" font-weight="bold" fill="{ACC}">KDA ×69</text>')
o.append(f'<text x="{lx+28}" y="{ly+37}" font-size="18" fill="{INK}">fixed-size state</text>')
ly2 = ly + 70
o.append(f'<rect x="{lx}" y="{ly2}" width="16" height="34" fill="#BFBFBF"/>')
o.append(f'<text x="{lx+28}" y="{ly2+15}" font-size="20" font-weight="bold" fill="{INK}">Gated MLA ×24</text>')
o.append(f'<text x="{lx+28}" y="{ly2+37}" font-size="18" fill="{INK}">KV cache grows</text>')
defs = ('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" '
        'markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#222222"/></marker></defs>')
o.insert(1, defs)
o.append('</svg>')
p = pathlib.Path(__file__).with_name("fig_k3_stack.svg"); p.write_text("\n".join(o)); print("wrote", p)
