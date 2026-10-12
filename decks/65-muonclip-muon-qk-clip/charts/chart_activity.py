"""chart_activity: share of K2 attention heads that triggered QK-Clip at least once.
Steps 0-70K: 12.7%; after 70K: all heads below tau, QK-Clip inactive (K2 App. D).
Token conversion with the 67M-token batch (derived): 70K steps = 4.7T tokens; 15.5T tokens = 231K steps."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

sc.setup()
fig, ax = sc.figure("right")
sc.bars(ax, ["Steps 0–70K\n(first ~4.7T tokens)", "Steps 70K–231K\n(remaining ~10.8T)"], [12.7, 0.0], ours=0,
        labels=["12.7%", "0%"], width=0.56)
ax.set_ylabel("Heads that hit QK-Clip (%)")
ax.set_ylim(0, 16); ax.set_yticks([0, 5, 10, 15])
sc.save(fig, __file__)
