"""Poster figures and extra tables from the notebook outputs.

Usage:  python poster_figures.py [outputs_dir] [cocoa_mt_test_dir]
Defaults: outputs/ and contrastive-controlled-mt/CoCoA-MT/test/en-de (as created by the notebook).

Writes to <outputs_dir>/poster/:
  fig_context.svg/.pdf/.png   RQ1+RQ2: share of Sie vs du per system and setting (uncontrolled)
  fig_control.svg/.pdf/.png   RQ3: share of Sie vs du per LLM and prompt
  table_main.csv              default formality, domain effect and translation quality per system
  table_control.csv           matched accuracy of prompt control and quality with/without control
  robustness.json             agreement between the two formality detectors
Figure sizes are the real print sizes on the A1 poster, so font sizes are true point sizes.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import sacrebleu

OUT = sys.argv[1] if len(sys.argv) > 1 else "outputs"
DATA = sys.argv[2] if len(sys.argv) > 2 else "contrastive-controlled-mt/CoCoA-MT/test/en-de"
POSTER = os.path.join(OUT, "poster")
os.makedirs(POSTER, exist_ok=True)

MM = 1 / 25.4
SIE, DU = "#2a78d6", "#eb6834"            # one colour language for the whole poster: blue = Sie, orange = du
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#dcdbd6", "#ffffff"
FONT = "Carlito"
plt.rcParams.update({"font.family": FONT, "font.size": 24, "svg.fonttype": "none",
                     "pdf.fonttype": 42, "axes.edgecolor": GRID, "axes.linewidth": 1})

DOMAINS = ["call_center", "telephony", "topical_chat"]
DOMAIN_NAMES = {"call_center": "Customer service", "telephony": "Phone conversation",
                "topical_chat": "Casual chat"}
SYSTEMS = ["opus-mt", "nllb-600m", "qwen2.5-1.5b", "qwen2.5-3b"]
SYSTEM_NAMES = {"opus-mt": "Opus-MT", "nllb-600m": "NLLB-200 600M",
                "qwen2.5-1.5b": "Qwen2.5 1.5B", "qwen2.5-3b": "Qwen2.5 3B"}
SYSTEM_TYPE = {"opus-mt": "bilingual NMT", "nllb-600m": "multilingual NMT",
               "qwen2.5-1.5b": "LLM", "qwen2.5-3b": "LLM"}

per = pd.read_csv(os.path.join(OUT, "per_sentence.csv"))
per["hyp"] = per["hyp"].fillna("")
tests = pd.read_csv(os.path.join(OUT, "stat_tests.csv"))

refs = {}
for d in DOMAINS:
    for k in ("formal", "informal"):
        with open(f"{DATA}/formality-control.test.{d}.en-de.{k}", encoding="utf-8") as f:
            for i, line in enumerate(f):
                refs[(d, i, k)] = line.strip()


def share(df):
    """% Sie and % du among segments with a detectable form of address (official scorer)."""
    m = df[df.label.isin(["FORMAL", "INFORMAL"])]
    f = 100 * (m.label == "FORMAL").mean()
    return f, 100 - f, len(m)


def chrf(system, cond, ref_kind):
    s = per[(per.system == system) & (per.condition == cond)]
    r = [[refs[(d, i, ref_kind)] for d, i in zip(s.domain, s.idx)]]
    return round(sacrebleu.corpus_chrf(s.hyp.tolist(), r).score, 1)


# ---------------- tables ----------------
rows = []
for s in SYSTEMS:
    t = tests[(tests.system == s) & (tests.condition == "neutral")].iloc[0]
    f, _, n = share(per[(per.system == s) & (per.condition == "neutral")])
    rows.append(dict(system=SYSTEM_NAMES[s], type=SYSTEM_TYPE[s], pct_sie=round(f, 1), n_matched=n,
                     binom_p=t.binom_p_vs_50pct, chi2=t.chi2, chi2_p=t.chi2_p, cramers_v=t.cramers_v,
                     chrF_vs_formal_ref=chrf(s, "neutral", "formal"),
                     chrF_vs_informal_ref=chrf(s, "neutral", "informal")))
table_main = pd.DataFrame(rows)
table_main.to_csv(os.path.join(POSTER, "table_main.csv"), index=False)

rows = []
for s in ["qwen2.5-1.5b", "qwen2.5-3b"]:
    for cond, target, ref_kind in [("formal", "FORMAL", "formal"), ("informal", "INFORMAL", "informal")]:
        sub = per[(per.system == s) & (per.condition == cond)]
        m = sub[sub.label.isin(["FORMAL", "INFORMAL"])]
        t = tests[(tests.system == s) & (tests.condition == cond)].iloc[0]
        rows.append(dict(system=SYSTEM_NAMES[s], prompt=cond, matched_accuracy=round(100 * (m.label == target).mean(), 1),
                         n_matched=len(m), domain_chi2_p=t.chi2_p, cramers_v=t.cramers_v,
                         chrF_controlled=chrf(s, cond, ref_kind), chrF_uncontrolled_same_ref=chrf(s, "neutral", ref_kind)))
table_control = pd.DataFrame(rows)
table_control.to_csv(os.path.join(POSTER, "table_control.csv"), index=False)

# exploratory: do single register cue words in the source shift the choice? (uncontrolled runs)
CUES = {"please": r"\bplease\b", "lol": r"\blol\b"}
rows = []
neutral = per[(per.condition == "neutral") & per.label.isin(["FORMAL", "INFORMAL"])]
for cue, pat in CUES.items():
    has = neutral.src.str.contains(pat, case=False, regex=True)
    for s in SYSTEMS:
        m = neutral.system == s
        rows.append(dict(cue=cue, system=SYSTEM_NAMES[s],
                         n_segments_with_cue=int(per[(per.system == s) & (per.condition == "neutral")]
                                                 .src.str.contains(pat, case=False, regex=True).sum()),
                         n_matched_with_cue=int((m & has).sum()),
                         pct_sie_with_cue=round(100 * (neutral[m & has].label == "FORMAL").mean(), 1),
                         pct_sie_without_cue=round(100 * (neutral[m & ~has].label == "FORMAL").mean(), 1)))
table_cues = pd.DataFrame(rows)
table_cues.to_csv(os.path.join(POSTER, "table_cues.csv"), index=False)

both = per[per.label.isin(["FORMAL", "INFORMAL"]) & per.label_lexical.isin(["FORMAL", "INFORMAL"])]
po = (both.label == both.label_lexical).mean()
p1, p2 = (both.label == "FORMAL").mean(), (both.label_lexical == "FORMAL").mean()
pe = p1 * p2 + (1 - p1) * (1 - p2)
robust = dict(n_both_labelled=len(both), agreement=round(po, 4), cohens_kappa=round((po - pe) / (1 - pe), 4),
              coverage_official=round(per.label.isin(["FORMAL", "INFORMAL"]).mean(), 4),
              coverage_lexical=round(per.label_lexical.isin(["FORMAL", "INFORMAL"]).mean(), 4))
with open(os.path.join(POSTER, "robustness.json"), "w") as f:
    json.dump(robust, f, indent=2)


# ---------------- figure helpers ----------------
def rounded_segment(ax, x0, width, y, height, color, round_left, round_right):
    """Bar segment with rounded outer ends only (square where segments meet)."""
    r = min(0.9, width / 2)
    ax.add_patch(FancyBboxPatch((x0, y - height / 2), width, height, boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=height / 6, linewidth=0, facecolor=color))
    # square off the inner ends
    if not round_left:
        ax.add_patch(plt.Rectangle((x0, y - height / 2), min(r, width), height, linewidth=0, facecolor=color))
    if not round_right:
        ax.add_patch(plt.Rectangle((x0 + width - min(r, width), y - height / 2), min(r, width), height,
                                   linewidth=0, facecolor=color))


def stacked_row(ax, y, pct_sie, height, gap=0.35, min_label=17):
    pct_du = 100 - pct_sie
    rounded_segment(ax, 0, max(pct_sie - gap / 2, 0.01), y, height, SIE, True, False)
    rounded_segment(ax, pct_sie + gap / 2, max(pct_du - gap / 2, 0.01), y, height, DU, False, True)
    if pct_sie >= min_label:
        ax.text(1.6, y, f"{pct_sie:.0f} % Sie", va="center", ha="left", color="white",
                fontsize=24, fontweight="bold")
    if pct_du >= min_label:
        ax.text(98.4, y, f"{pct_du:.0f} % du", va="center", ha="right", color=INK,
                fontsize=24, fontweight="bold")


def style_axis(ax, ymax, groups):
    ax.set_xlim(0, 100)
    ax.set_ylim(ymax, -0.2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    for y0, y1 in groups:                       # 50 % reference line only behind each group's bars
        ax.plot([50, 50], [y0, y1], color=MUTED, lw=1.2, zorder=0, solid_capstyle="butt")
    ax.set_xticks([0, 25, 50, 75, 100], ["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.tick_params(axis="x", colors=INK2, labelsize=24, length=0, pad=10)
    ax.tick_params(axis="y", length=0, labelsize=24, colors=INK, pad=14)


# ---------------- Figure 1: context (uncontrolled) ----------------
fig, ax = plt.subplots(figsize=(318 * MM, 226 * MM))
y, yticks, ylabels, group_y, groups = 0.0, [], [], [], []
BAR, STEP, GROUP_GAP = 0.72, 1.0, 1.05
for s in SYSTEMS:
    group_y.append(y)
    y += 0.85                                   # room for the system name above its group
    y_first = y
    for d in DOMAINS:
        f, _, _ = share(per[(per.system == s) & (per.condition == "neutral") & (per.domain == d)])
        stacked_row(ax, y, f, BAR)
        yticks.append(y); ylabels.append(DOMAIN_NAMES[d])
        y += STEP
    groups.append((y_first - BAR / 2 - 0.12, y - STEP + BAR / 2 + 0.12))
    y += GROUP_GAP - STEP + 0.35
style_axis(ax, y - 0.55, groups)
ax.set_yticks(yticks, ylabels)
for s, gy in zip(SYSTEMS, group_y):
    t = tests[(tests.system == s) & (tests.condition == "neutral")].iloc[0]
    v = np.sqrt(t.chi2 / t.n_matched)           # Cramér's V for a k x 2 table
    ax.text(0, gy + 0.18, SYSTEM_NAMES[s], ha="left", va="center", fontsize=26, fontweight="bold", color=INK)
    ax.text(100, gy + 0.18, f"{SYSTEM_TYPE[s]} · V = {v:.2f}", ha="right", va="center",
            fontsize=24, color=INK2)
fig.subplots_adjust(left=0.235, right=0.96, top=0.99, bottom=0.075)
for ext in ("svg", "pdf", "png"):
    fig.savefig(os.path.join(POSTER, f"fig_context.{ext}"), dpi=300, facecolor=SURFACE)
plt.close(fig)

# ---------------- Figure 2: prompt control (LLMs) ----------------
PROMPT_NAMES = {"informal": "“Use du”", "neutral": "No instruction", "formal": "“Use Sie”"}
fig, ax = plt.subplots(figsize=(222 * MM, 128 * MM))
y, yticks, ylabels, group_y, groups = 0.0, [], [], [], []
for s in ["qwen2.5-1.5b", "qwen2.5-3b"]:
    group_y.append(y)
    y += 0.85
    y_first = y
    for cond in ["informal", "neutral", "formal"]:
        f, _, _ = share(per[(per.system == s) & (per.condition == cond)])
        stacked_row(ax, y, f, BAR)
        yticks.append(y); ylabels.append(PROMPT_NAMES[cond])
        y += STEP
    groups.append((y_first - BAR / 2 - 0.12, y - STEP + BAR / 2 + 0.12))
    y += GROUP_GAP - STEP + 0.35
style_axis(ax, y - 0.55, groups)
ax.set_yticks(yticks, ylabels)
for s, gy in zip(["qwen2.5-1.5b", "qwen2.5-3b"], group_y):
    ax.text(0, gy + 0.18, SYSTEM_NAMES[s], ha="left", va="center", fontsize=26, fontweight="bold", color=INK)
fig.subplots_adjust(left=0.25, right=0.95, top=0.985, bottom=0.13)
for ext in ("svg", "pdf", "png"):
    fig.savefig(os.path.join(POSTER, f"fig_control.{ext}"), dpi=300, facecolor=SURFACE)
plt.close(fig)

print(table_main.to_string(index=False))
print(table_control.to_string(index=False))
print(robust)
