"""fig_costs.svg: what KDA's serial fixed-size state demands from the system (K3 Sec. 5.1, 5.4.1, 5.4.2)."""
import pathlib
ACC, ACC_F, INK, GRAY, GRAY_F, RED, RED_F = "#1F4E79", "#DEEBF7", "#222222", "#595959", "#F2F2F2", "#C00000", "#FBE5E5"
W, H = 480, 290
o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H}pt" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
     '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#222222"/></marker></defs>']
def t(x, y, s, size=18, w="normal", fill=INK, anchor="middle", style="normal"):
    o.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}" font-style="{style}">{s}</text>')
def box(x, y, w, h, f, s, sw=1.5, rx=5):
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{f}" stroke="{s}" stroke-width="{sw}"/>')
box(6, 104, 150, 82, RED_F, RED, 2.5, 6)
t(81, 138, "Serial state", 19, "bold", RED)
t(81, 164, "fixed size", 17)
items = [("FlashKDA kernel", "overlap (deck 08)"), ("KCP all-gather", "across GPUs (deck 69)"),
         ("State checkpoints", "prefix cache"), ("Input replay", "speculative decode")]
for k, (a, b) in enumerate(items):
    y = 6 + k * 71
    box(214, y, 260, 60, ACC_F, ACC, 2)
    t(344, y + 26, a, 18, "bold", ACC)
    t(344, y + 49, b, 16)
    o.append(f'<path d="M158,145 C186,145 186,{y+30} 212,{y+30}" fill="none" stroke="{INK}" stroke-width="2" marker-end="url(#a)"/>')
o.append('</svg>')
p = pathlib.Path(__file__).with_name("fig_costs.svg"); p.write_text("\n".join(o)); print("wrote", p)
