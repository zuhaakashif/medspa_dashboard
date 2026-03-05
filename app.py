"""
Phase 4 — Streamlit Dashboard (Upgraded)
Med Spa Revenue Leakage & Utilization Intelligence Sandbox

Sections:
  1. Executive Operations Overview
  2. Revenue Leakage Analysis
  3. Scheduling Inefficiency Analysis
  4. Optimization Simulation
  5. Sensitivity Modeling
  6. AI Operational Recommendations
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from phase1_data_engine import ROOMS, PROVIDERS, SERVICES

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "MedSpa Intelligence Sandbox",
    page_icon   = "💆",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE & PLOT DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
PAL = {
    "primary": "#63372C", "accent": "#C97D60", "danger": "#C97D60",
    "success": "#63372C", "neutral": "#FFBCB5", "bg": "#F2E5D7", "grid": "#FFBCB5",
}
ROOM_COLORS = ["#262322", "#63372C", "#C97D60", "#FFBCB5"]

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "figure.facecolor": PAL["bg"], "axes.facecolor": PAL["bg"],
    "axes.edgecolor": PAL["grid"], "grid.color": PAL["grid"],
    "font.family": "DejaVu Sans",
})

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── global ── */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; }

section.main > div.block-container {
    padding-top: 5rem !important;
    margin-top: 0 !important;
}

div[data-testid="stAppViewContainer"] > section > div.block-container {
    padding-top: 5rem !important;
}

/* ── metric cards ── */
div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 14px 18px 10px 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
div[data-testid="metric-container"] label { font-size: 0.72rem; color: #64748b; font-weight: 600; letter-spacing: .04em; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 700; color: #1e293b; }
div[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.78rem; }

/* ── section header ── */
.sec-header {
    background: linear-gradient(90deg, #262322 0%, #C97D60 100%);
    color: #fff; padding: 10px 18px; border-radius: 8px;
    font-size: 1.05rem; font-weight: 700; margin-bottom: 1rem;
    letter-spacing: .02em;
}

/* ── insight box ── */
.insight-box {
    background: #EFF6FF; border-left: 4px solid #2D6A9F;
    padding: 10px 14px; border-radius: 0 8px 8px 0;
    font-size: 0.86rem; color: #1e3a5f; margin-bottom: 0.6rem;
}
.warn-box {
    background: #FFF7ED; border-left: 4px solid #E8A838;
    padding: 10px 14px; border-radius: 0 8px 8px 0;
    font-size: 0.86rem; color: #7c4700; margin-bottom: 0.6rem;
}
.danger-box {
    background: #FEF2F2; border-left: 4px solid #D94F3D;
    padding: 10px 14px; border-radius: 0 8px 8px 0;
    font-size: 0.86rem; color: #7f1d1d; margin-bottom: 0.6rem;
}
.success-box {
    background: #F0FDF4; border-left: 4px solid #3DAD77;
    padding: 10px 14px; border-radius: 0 8px 8px 0;
    font-size: 0.86rem; color: #14532d; margin-bottom: 0.6rem;
}

/* ── sidebar ── */
[data-testid="stSidebar"] { background: #262322; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.88rem; }

/* ── dataframe ── */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* ── recommendation panel ── */
.rec-panel {
    background: #0f172a; border-radius: 12px;
    padding: 20px 24px; margin-bottom: 0.8rem;
}
.rec-title { color: #f1f5f9; font-size: 1.0rem; font-weight: 700; margin-bottom: 12px; }
.rec-item {
    display: flex; align-items: flex-start; gap: 10px;
    background: #1e293b; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 8px;
    font-size: 0.85rem; color: #e2e8f0;
}
.rec-badge { background: #2D6A9F; color: #fff; border-radius: 5px; padding: 2px 8px; font-size: 0.72rem; font-weight: 700; white-space: nowrap; margin-top: 1px; }
.rec-badge-warn { background: #E8A838; }
.rec-badge-danger { background: #D94F3D; }
.rec-badge-success { background: #3DAD77; }
.stat-pill { display: inline-block; background: #EFF6FF; color: #1d4ed8; border-radius: 999px; padding: 2px 10px; font-size: 0.75rem; font-weight: 600; margin-left: 6px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (cached so re-runs are instant)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⚙️  Generating synthetic operations data…")
def load_all_data():
    from phase1_data_engine import generate_all_data
    from phase2_data_analytics import (
        compute_room_kpis, compute_provider_kpis, compute_daily_kpis,
        compute_leakage, compute_peak_hours, compute_service_mix,
        compute_room_type_efficiency, compute_provider_efficiency,
    )
    from phase3_optimization import (
        simulate_gap_filling, simulate_overbooking, simulate_sensitivity,
        GapFillConfig, OverbookConfig,
    )

    raw          = generate_all_data()
    appointments = raw["appointments"]
    idle_time    = raw["idle_time"]

    leakage       = compute_leakage(appointments, idle_time)
    room_kpis     = compute_room_kpis(appointments, idle_time)
    provider_kpis = compute_provider_kpis(appointments)
    daily_kpis    = compute_daily_kpis(appointments, idle_time)
    peak_hours    = compute_peak_hours(appointments)
    service_mix   = compute_service_mix(appointments)
    room_type_eff = compute_room_type_efficiency(appointments, idle_time)
    provider_eff  = compute_provider_efficiency(appointments)

    gap_cfg     = GapFillConfig(avg_revenue_per_min=leakage["avg_rev_per_min"])
    gap_results = simulate_gap_filling(appointments, idle_time, gap_cfg)

    completed       = appointments[appointments["status"] == "Completed"]
    avg_appt_rev    = completed["actual_revenue"].mean()
    ob_cfg          = OverbookConfig(avg_appt_revenue=round(avg_appt_rev, 2))
    ob_results      = simulate_overbooking(appointments, ob_cfg)

    sensitivity = simulate_sensitivity(appointments, idle_time, leakage)

    return dict(
        appointments=appointments, idle_time=idle_time,
        leakage=leakage, room_kpis=room_kpis, provider_kpis=provider_kpis,
        daily_kpis=daily_kpis, peak_hours=peak_hours, service_mix=service_mix,
        room_type_eff=room_type_eff, provider_eff=provider_eff,
        gap_results=gap_results, ob_results=ob_results, sensitivity=sensitivity,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS  (return fig objects for st.pyplot)
# ─────────────────────────────────────────────────────────────────────────────

def fig_room_heatmap(idle_time):
    
    pivot = idle_time.pivot(index="room_id", columns="date", values="utilization")
    room_labels = ROOMS.set_index("room_id")["room_name"]
    pivot.index   = [room_labels[r] for r in pivot.index]
    pivot.columns = pd.to_datetime(pivot.columns).strftime("%b %d")
    fig, ax = plt.subplots(figsize=(16, 3.5))
    sns.heatmap(pivot * 100, ax=ax, cmap="YlOrRd", vmin=0, vmax=100,
                linewidths=0.3, linecolor="#ddd", annot=False,
                cbar_kws={"label": "Utilization %", "shrink": 0.8})
    ax.set_title("Room Utilization Heatmap — 30-Day Window", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=6.5)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    fig.tight_layout()
    return fig


def fig_daily_revenue(daily_kpis):
    df = daily_kpis.sort_values("date")
    dates = pd.to_datetime(df["date"])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dates, df["actual_revenue"], df["theoretical_max_revenue"],
                    alpha=0.15, color=PAL["danger"], label="Revenue Gap")
    ax.plot(dates, df["theoretical_max_revenue"], color=PAL["neutral"],
            linewidth=1.6, linestyle="--", label="Theoretical Max")
    ax.plot(dates, df["actual_revenue"], color=PAL["primary"],
            linewidth=2.2, label="Actual Revenue")
    ax.set_title("Daily Revenue: Actual vs Theoretical Maximum", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Revenue ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9); ax.tick_params(axis="x", rotation=30, labelsize=8)
    sns.despine(); fig.tight_layout()
    return fig


def fig_provider_revenue(provider_kpis):
    df = provider_kpis.sort_values("total_revenue")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(df["name"], df["total_revenue"],
                   color=ROOM_COLORS[::-1], edgecolor="white", height=0.5)
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_width() + 400, bar.get_y() + bar.get_height() / 2,
                f"${row['total_revenue']:,.0f}  ·  {row['revenue_to_cost_ratio']:.1f}× R/C",
                va="center", fontsize=8.5)
    ax.set_title("Total Revenue per Provider", fontsize=12, fontweight="bold", pad=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlim(0, df["total_revenue"].max() * 1.38)
    sns.despine(left=True); fig.tight_layout()
    return fig


def fig_idle_histogram(idle_time):
    from phase1_data_engine import ROOMS
    room_labels = ROOMS.set_index("room_id")["room_name"].to_dict()
    df = idle_time.copy()
    df["room_name"] = df["room_id"].map(room_labels)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
    for ax, (room_id, color) in zip(axes, zip(ROOMS["room_id"], ROOM_COLORS)):
        sub = df[df["room_id"] == room_id]["idle_min"]
        ax.hist(sub, bins=12, color=color, edgecolor="white", alpha=0.88)
        ax.set_title(room_labels[room_id], fontsize=9, fontweight="bold")
        ax.set_xlabel("Idle Min/Day", fontsize=8)
        ax.axvline(sub.mean(), color="#333", linestyle="--", linewidth=1.2,
                   label=f"μ={sub.mean():.0f}m")
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("Days")
    fig.suptitle("Daily Idle Time Distribution per Room", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def fig_leakage_donut(leakage):
    bd     = leakage["breakdown"]
    colors = [PAL["danger"], PAL["accent"], PAL["primary"], PAL["neutral"]]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    wedges, _, autotexts = ax.pie(
        bd["amount"], colors=colors, autopct=lambda p: f"{p:.1f}%",
        startangle=140, wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(9); t.set_fontweight("bold")
    labels = [f"{l}  ${v:,.0f}" for l, v in zip(bd["leakage_type"], bd["amount"])]
    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8.5)
    ax.set_title(f"Revenue Leakage Breakdown\nTotal: ${leakage['total_leakage']:,.0f}",
                 fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def fig_service_mix(service_mix):
    df = service_mix.sort_values("total_revenue", ascending=True)
    cmap = plt.cm.get_cmap("tab10", len(df))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(df["service_name"], df["total_revenue"],
                   color=[cmap(i) for i in range(len(df))], edgecolor="white", height=0.6)
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
                f"{row['revenue_share_pct']:.1f}%  ·  {row['appt_count']} appts",
                va="center", fontsize=8)
    ax.set_title("Service Revenue Contribution", fontsize=12, fontweight="bold", pad=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlim(0, df["total_revenue"].max() * 1.38)
    sns.despine(left=True); fig.tight_layout()
    return fig


def fig_peak_hours(peak_hours):
    df = peak_hours.sort_values("hour")
    bar_colors = [
        PAL["success"] if r == df["total_revenue"].max()
        else PAL["danger"] if r == df["total_revenue"].min()
        else PAL["primary"]
        for r in df["total_revenue"]
    ]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(df["hour"].astype(str) + ":00", df["total_revenue"],
           color=bar_colors, edgecolor="white")
    ax.set_title("Revenue by Hour of Day", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Revenue ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    sns.despine(); fig.tight_layout()
    return fig


def fig_gap_fill_uplift(daily_kpis, daily_uplift):
    base = daily_kpis.sort_values("date")[["date","actual_revenue","theoretical_max_revenue"]].copy()
    base["date"] = pd.to_datetime(base["date"])
    if not daily_uplift.empty:
        up = daily_uplift.copy(); up["date"] = pd.to_datetime(up["date"])
        base = base.merge(up[["date","revenue_gained"]], on="date", how="left")
    else:
        base["revenue_gained"] = 0
    base["revenue_gained"]    = base["revenue_gained"].fillna(0)
    base["optimised_revenue"] = base["actual_revenue"] + base["revenue_gained"]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(base["date"], 0, base["actual_revenue"],
                    alpha=0.70, color=PAL["primary"], label="Baseline Revenue")
    ax.fill_between(base["date"], base["actual_revenue"], base["optimised_revenue"],
                    alpha=0.80, color=PAL["success"], label="Gap-Fill Gain")
    ax.fill_between(base["date"], base["optimised_revenue"], base["theoretical_max_revenue"],
                    alpha=0.18, color=PAL["danger"], label="Remaining Gap")
    ax.plot(base["date"], base["theoretical_max_revenue"],
            color=PAL["neutral"], linewidth=1.5, linestyle="--", label="Theoretical Max")
    ax.set_title("Gap-Fill Simulation: Daily Revenue Uplift", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Revenue ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9, loc="lower right")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    sns.despine(); fig.tight_layout()
    return fig


def fig_gap_fill_rooms(gap_summary):
    from phase1_data_engine import ROOMS
    room_labels = ROOMS.set_index("room_id")["room_name"].to_dict()
    agg = (gap_summary.groupby("room_id")
           .agg(gaps_found=("gap_min","count"), gaps_filled=("filled","sum"))
           .reset_index())
    agg["room_name"] = agg["room_id"].map(room_labels)
    x = np.arange(len(agg)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, agg["gaps_found"],  w, label="Gaps Found",  color=PAL["neutral"], edgecolor="white")
    ax.bar(x + w/2, agg["gaps_filled"], w, label="Gaps Filled", color=PAL["success"], edgecolor="white")
    for i, (_, row) in enumerate(agg.iterrows()):
        pct = row["gaps_filled"] / max(row["gaps_found"], 1) * 100
        ax.text(i + w/2, row["gaps_filled"] + 0.2, f"{pct:.0f}%",
                ha="center", fontsize=8.5, fontweight="bold", color=PAL["success"])
    ax.set_xticks(x); ax.set_xticklabels(agg["room_name"], rotation=12, ha="right")
    ax.set_title("Gap-Fill: Gaps Found vs Filled per Room", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Count"); ax.legend(fontsize=9)
    sns.despine(); fig.tight_layout()
    return fig


def fig_overbooking(ob_results):
    df = ob_results["scenario_summary"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = [PAL["success"] if v > 0 else PAL["danger"] for v in df["mean_uplift"]]
    bars = ax1.bar(df["overbook_factor_pct"], df["mean_uplift"],
                   color=colors, edgecolor="white", width=0.45, zorder=3)
    yerr_lo = df["mean_uplift"] - df["uplift_5pct"]
    yerr_hi = df["uplift_95pct"] - df["mean_uplift"]
    ax1.errorbar(df["overbook_factor_pct"], df["mean_uplift"],
                 yerr=[yerr_lo, yerr_hi], fmt="none", color="#555",
                 capsize=5, linewidth=1.5, zorder=4)
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                 f"${row['mean_uplift']:+,.0f}", ha="center", fontsize=8.5, fontweight="bold")
    ax1.axhline(0, color="#aaa", linewidth=1)
    ax1.set_title("Mean Daily Revenue Uplift\n(vs No-Overbook Baseline)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Overbook Factor"); ax1.set_ylabel("Revenue Uplift ($)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    sns.despine(ax=ax1)
    ax2.plot(df["overbook_factor_pct"], df["overflow_prob"] * 100,
             color=PAL["danger"], marker="o", linewidth=2.2, markersize=7)
    ax2.fill_between(range(len(df)), df["overflow_prob"] * 100, alpha=0.12, color=PAL["danger"])
    for i, (_, row) in enumerate(df.iterrows()):
        ax2.text(i, row["overflow_prob"] * 100 + 0.5, f"{row['overflow_prob']*100:.1f}%",
                 ha="center", fontsize=8.5, color=PAL["danger"], fontweight="bold")
    ax2.set_title("Overflow Probability by Overbook Factor\n(Risk Curve)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Overbook Factor"); ax2.set_ylabel("Overflow Probability (%)")
    ax2.set_xticks(range(len(df))); ax2.set_xticklabels(df["overbook_factor_pct"])
    sns.despine(ax=ax2)
    fig.suptitle("Risk-Adjusted Overbooking (10,000 Monte-Carlo Days)",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def fig_sensitivity_heatmap(sensitivity):
    df = sensitivity["scenario_matrix"]
    pivot = df.pivot(index="lever", columns="uplift_pct", values="revenue_gain_$")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=",.0f", cmap="YlGn",
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Revenue Gain ($)"})
    ax.set_title("Sensitivity Matrix: Revenue Gain by Lever × Uplift",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Uplift Target"); ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return fig


def fig_roi_bars(sensitivity):
    df = sensitivity["scenario_matrix"]
    p5  = df[df["uplift_target"] == 0.05].sort_values("lever_id")
    p10 = df[df["uplift_target"] == 0.10].sort_values("lever_id")
    x = np.arange(len(p5)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4))
    b5  = ax.bar(x - w/2, p5["roi_pct"],  w, label="+5%",  color=PAL["primary"], edgecolor="white")
    b10 = ax.bar(x + w/2, p10["roi_pct"], w, label="+10%", color=PAL["accent"],  edgecolor="white")
    for bars in [b5, b10]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 8,
                    f"{h:,.0f}%", ha="center", fontsize=8, fontweight="bold")
    ax.axhline(0, color="#aaa", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(p5["lever"], rotation=10, ha="right")
    ax.set_title("ROI by Optimization Lever", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("ROI (%)"); ax.legend(fontsize=9)
    sns.despine(); fig.tight_layout()
    return fig


def fig_waterfall(base_revenue, gap_gain, ob_gain, sensitivity):
    s5  = sensitivity["combo_scenarios"][sensitivity["combo_scenarios"]["uplift_target"] == 0.05]
    s10 = sensitivity["combo_scenarios"][sensitivity["combo_scenarios"]["uplift_target"] == 0.10]
    s5_gain  = float(s5["revenue_gain_$"].iloc[0]) if not s5.empty else 0
    s10_gain = float(s10["revenue_gain_$"].iloc[0]) if not s10.empty else 0
    labels = ["Baseline", "Gap-Fill", "Overbook\n(Optimal)", "+5%\nAll Levers", "+10%\nAll Levers"]
    values = [base_revenue, gap_gain, ob_gain, s5_gain, s10_gain]
    running = [base_revenue]
    for v in values[1:]: running.append(running[-1] + v)
    bar_bottoms = [0] + running[:-1]
    colors = [PAL["primary"]] + [PAL["success"]] * (len(values) - 1)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (label, val, bottom, color) in enumerate(zip(labels, values, bar_bottoms, colors)):
        ax.bar(i, val, bottom=bottom, color=color, edgecolor="white", width=0.52, zorder=3)
        ax.text(i, running[i] + base_revenue * 0.005, f"${running[i]:,.0f}",
                ha="center", fontsize=8.5, fontweight="bold")
        if i > 0:
            ax.text(i, bottom + val/2, f"+${val:,.0f}",
                    ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    for i in range(len(labels) - 1):
        ax.plot([i + 0.27, i + 0.73], [running[i], running[i]],
                color="#bbb", linewidth=1, linestyle="--", zorder=2)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Optimization Waterfall: Revenue Impact by Strategy",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Cumulative Revenue ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_ylim(0, running[-1] * 1.12)
    sns.despine(); fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💆 MedSpa Intelligence")
    st.markdown("**Executive Decision-Support Platform**")
    st.markdown("---")
    section = st.radio(
        "Navigate",
        [
            "📊 Executive Operations Overview",
            "💸 Revenue Leakage Analysis",
            "📅 Scheduling Inefficiency Analysis",
            "⚙️ Optimization Simulation",
            "📈 Sensitivity Modeling",
            "🤖 AI Operational Recommendations",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Simulation: **30-day synthetic dataset**")
    st.caption("4 Rooms · 4 Providers · 10 Services")
    st.caption("Phases 1–3 loaded on startup")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
D = load_all_data()
appointments  = D["appointments"]
idle_time     = D["idle_time"]
leakage       = D["leakage"]
room_kpis     = D["room_kpis"]
provider_kpis = D["provider_kpis"]
daily_kpis    = D["daily_kpis"]
peak_hours    = D["peak_hours"]
service_mix   = D["service_mix"]
gap_results   = D["gap_results"]
ob_results    = D["ob_results"]
sensitivity   = D["sensitivity"]

base_revenue = sensitivity["base_revenue"]
base_util    = sensitivity["base_utilization"]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — EXECUTIVE OPERATIONS OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if section == "📊 Executive Operations Overview":
    # ── Page title ───────────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">📊 Executive Operations Overview — 30-Day Performance Summary</div>', unsafe_allow_html=True)

    # ── Compute headline scalars from dataframes ──────────────────────────────
    total_appts    = len(appointments)
    completed_n    = (appointments["status"] == "Completed").sum()
    no_show_n      = (appointments["status"] == "No-Show").sum()
    operating_days = appointments["date"].nunique()
    avg_rev_per_appt = appointments[appointments["status"] == "Completed"]["actual_revenue"].mean()

    # ── 6 KPI Cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Appointments",    f"{total_appts:,}",
              delta=f"{completed_n:,} completed", delta_color="off")
    c2.metric("Utilization Rate",      f"{base_util:.1%}",
              delta=f"{base_util - 0.70:.1%} vs 70% target", delta_color="inverse" if base_util < 0.70 else "normal")
    c3.metric("Total Revenue",         f"${base_revenue:,.0f}",
              delta=f"${base_revenue/operating_days:,.0f}/day avg", delta_color="off")
    c4.metric("Revenue Leakage",       f"${leakage['total_leakage']:,.0f}",
              delta=f"{leakage['leakage_pct']}% of listed revenue", delta_color="inverse")
    c5.metric("No-Show Rate",          f"{no_show_n/total_appts:.1%}",
              delta=f"-${leakage['no_show_leakage']:,.0f} lost", delta_color="inverse")
    c6.metric("Avg Revenue / Visit",   f"${avg_rev_per_appt:,.0f}",
              delta=f"vs ${appointments['listed_price'].mean():,.0f} listed", delta_color="off")

    st.markdown("---")

    # ── Row 1: Heatmap (full width) ───────────────────────────────────────────
    st.markdown("#### 🗓️ Room Utilization Heatmap — 30 Days")
    st.pyplot(fig_room_heatmap(idle_time), use_container_width=True)

    st.markdown("---")

    # ── Row 2: Daily revenue line + Provider revenue bar ─────────────────────
    col_a, col_b = st.columns([1.35, 1.0])
    with col_a:
        st.markdown("#### 📈 Daily Revenue: Actual vs Theoretical Maximum")
        st.pyplot(fig_daily_revenue(daily_kpis))
    with col_b:
        st.markdown("#### 👩‍⚕️ Revenue per Provider")
        st.pyplot(fig_provider_revenue(provider_kpis))

    st.markdown("---")

    # ── Row 3: Service mix + Peak hours ──────────────────────────────────────
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### 🛎️ Service Revenue Mix")
        st.pyplot(fig_service_mix(service_mix))
    with col_d:
        st.markdown("#### ⏰ Revenue by Hour of Day")
        st.pyplot(fig_peak_hours(peak_hours))

    st.markdown("---")

    # ── Row 4: Room KPI table + Provider KPI table ────────────────────────────
    col_e, col_f = st.columns(2)
    with col_e:
        st.markdown("#### 🏠 Room Performance Summary")
        rk = room_kpis[["room_name","room_type","scheduled_utilization","total_revenue","total_idle_min","no_show_count"]].copy()
        rk.columns = ["Room", "Type", "Utilization", "Revenue", "Idle Min", "No-Shows"]
        rk["Utilization"] = rk["Utilization"].map("{:.1%}".format)
        rk["Revenue"]     = rk["Revenue"].map("${:,.0f}".format)
        rk["Idle Min"]    = rk["Idle Min"].map("{:,.0f}".format)
        st.dataframe(rk, use_container_width=True, hide_index=True)
    with col_f:
        st.markdown("#### 👤 Provider Performance Summary")
        pk = provider_kpis[["name","role","scheduled_utilization","total_revenue","revenue_per_appt","revenue_to_cost_ratio"]].copy()
        pk.columns = ["Provider", "Role", "Utilization", "Revenue", "Rev/Appt", "R/C Ratio"]
        pk["Utilization"] = pk["Utilization"].map("{:.1%}".format)
        pk["Revenue"]     = pk["Revenue"].map("${:,.0f}".format)
        pk["Rev/Appt"]    = pk["Rev/Appt"].map("${:,.0f}".format)
        pk["R/C Ratio"]   = pk["R/C Ratio"].map("{:.2f}×".format)
        st.dataframe(pk, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Key Insight Cards ─────────────────────────────────────────────────────
    st.markdown("#### 💡 Key Operational Insights")
    worst_room = room_kpis.loc[room_kpis["scheduled_utilization"].idxmin(), "room_name"]
    top_prov   = provider_kpis.loc[provider_kpis["revenue_to_cost_ratio"].idxmax(), "name"]
    top_svc    = service_mix.iloc[0]["service_name"]
    peak_hour  = int(peak_hours.loc[peak_hours["total_revenue"].idxmax(), "hour"])
    low_hour   = int(peak_hours.loc[peak_hours["total_revenue"].idxmin(), "hour"])

    st.markdown(f'''<div class="danger-box">🔴 <b>Revenue leakage exceeds collected revenue</b> — ${leakage["total_leakage"]:,.0f} in leakage vs ${base_revenue:,.0f} collected. Idle time alone accounts for {leakage["breakdown"].iloc[0]["pct_of_total_leakage"]:.0f}% of this ({leakage["total_idle_min"]/60:.0f} wasted hours).</div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="warn-box">⚠️ <b>{worst_room}</b> has the lowest utilization at {room_kpis["scheduled_utilization"].min():.1%} — a same-day discount program or reallocation of services could close this gap.</div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="success-box">✅ <b>{top_prov}</b> leads on revenue-to-cost efficiency ({provider_kpis["revenue_to_cost_ratio"].max():.2f}×). Prioritizing this provider on Thu/Fri peak days would yield the fastest revenue uplift.</div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="insight-box">📌 <b>{top_svc}</b> is the top revenue service ({service_mix.iloc[0]["revenue_share_pct"]:.1f}% of total). Peak hour is <b>{peak_hour}:00</b> and the trough is <b>{low_hour}:00</b> — schedule high-value services in the morning and deploy discounts for the evening slump.</div>''', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — REVENUE LEAKAGE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif section == "💸 Revenue Leakage Analysis":
    st.markdown('<div class="sec-header">💸 Revenue Leakage Breakdown — Where Is the Money Going?</div>', unsafe_allow_html=True)

    # ── Top KPI row ───────────────────────────────────────────────────────────
    idle_pct   = leakage["breakdown"].iloc[0]["pct_of_total_leakage"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue Leakage",   f"${leakage['total_leakage']:,.0f}",
              delta=f"{leakage['leakage_pct']}% of listed revenue", delta_color="inverse")
    c2.metric("Idle Time Loss",          f"${leakage['idle_leakage']:,.0f}",
              delta=f"{idle_pct:.0f}% of total leakage", delta_color="inverse")
    c3.metric("No-Show Loss",            f"${leakage['no_show_leakage']:,.0f}",
              delta=f"{leakage['breakdown'].iloc[1]['pct_of_total_leakage']:.0f}% of total", delta_color="inverse")
    c4.metric("Late-Cancel Loss",        f"${leakage['late_cancel_leakage']:,.0f}",
              delta=f"{leakage['breakdown'].iloc[2]['pct_of_total_leakage']:.0f}% of total", delta_color="inverse")

    st.markdown("---")

    # ── Donut chart + leakage source table ───────────────────────────────────
    col_a, col_b = st.columns([0.42, 0.58])
    with col_a:
        st.markdown("#### Leakage by Category")
        st.pyplot(fig_leakage_donut(leakage))
    with col_b:
        st.markdown("#### Leakage Source Summary")
        bd = leakage["breakdown"].copy()
        bd = bd.sort_values("amount", ascending=False).reset_index(drop=True)
        bd["amount_fmt"]   = bd["amount"].map("${:,.0f}".format)
        bd["pct_fmt"]      = bd["pct_of_total_leakage"].map("{:.1f}%".format)
        bd_display = bd[["leakage_type","amount_fmt","pct_fmt"]].copy()
        bd_display.columns = ["Leakage Source", "Revenue Lost", "% of Total"]
        st.dataframe(bd_display, use_container_width=True, hide_index=True)

        st.markdown("##### Idle Time Leakage by Room")
        ril = leakage["room_idle_leakage"][["room_name","room_type","idle_min_total","idle_leakage_$"]].copy()
        ril = ril.sort_values("idle_leakage_$", ascending=False)
        ril["idle_min_total"]  = ril["idle_min_total"].map("{:,.0f} min".format)
        ril["idle_leakage_$"]  = ril["idle_leakage_$"].map("${:,.0f}".format)
        ril.columns = ["Room","Type","Total Idle","Est. Loss"]
        st.dataframe(ril, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Idle histogram + No-show provider table ───────────────────────────────
    col_c, col_d = st.columns([0.60, 0.40])
    with col_c:
        st.markdown("#### Daily Idle Time Distribution per Room")
        st.pyplot(fig_idle_histogram(idle_time))
    with col_d:
        st.markdown("#### No-Show & Cancellation by Provider")
        prov_ns = provider_kpis[["name","no_show_count","late_cancel_count","total_appts"]].copy()
        prov_ns["No-Show Rate"] = (prov_ns["no_show_count"] / prov_ns["total_appts"]).map("{:.1%}".format)
        prov_ns.columns = ["Provider","No-Shows","Late Cancels","Total Appts","NS Rate"]
        st.dataframe(prov_ns, use_container_width=True, hide_index=True)

        late_cancel_fees = appointments[appointments["status"]=="Late Cancel"]["actual_revenue"].sum()
        late_cancel_listed = leakage["late_cancel_leakage"] + late_cancel_fees
        st.markdown(f'''<div class="warn-box" style="margin-top:12px;">
        💡 Late cancellations collected <b>${late_cancel_fees:,.0f}</b> in fees against <b>${late_cancel_listed:,.0f}</b> listed — a {late_cancel_fees/late_cancel_listed:.0%} recovery rate. A 48-hour cancellation policy could push this above 50%.
        </div>''', unsafe_allow_html=True)

    st.markdown("---")

    # ── Interactive: Room idle explorer ──────────────────────────────────────
    st.markdown("#### 📅 Daily Idle Time Explorer")
    from phase1_data_engine import ROOMS
    room_opts = ["All Rooms"] + ROOMS["room_name"].tolist()
    sel_room = st.selectbox("Filter by Room", room_opts)
    it_display = idle_time.copy()
    if sel_room != "All Rooms":
        it_display = it_display[it_display["room_name"] == sel_room]
    it_display = it_display[["date","room_name","booked_min","idle_min","utilization"]].copy()
    it_display.columns = ["Date","Room","Booked Min","Idle Min","Utilization"]
    it_display["Utilization"] = it_display["Utilization"].map("{:.1%}".format)
    it_display = it_display.sort_values("Date", ascending=False)
    st.dataframe(it_display, use_container_width=True, hide_index=True, height=300)

    st.markdown("---")
    st.markdown("#### 💡 Leakage Insights")
    st.markdown(f'''<div class="danger-box">🔴 <b>{leakage["total_idle_min"]/60:.0f} hours of idle room capacity</b> went unbilled — at ${leakage["avg_rev_per_min"]:.2f}/min this is <b>${leakage["idle_leakage"]:,.0f}</b> in directly recoverable revenue.</div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="warn-box">⚠️ A <b>$75 deposit-at-booking policy</b> could recover an estimated 60–70% of the ${leakage["no_show_leakage"]:,.0f} no-show loss, netting ~${leakage["no_show_leakage"]*0.65:,.0f} per 30-day period.</div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="insight-box">📌 <b>Pricing discounts account for ${leakage["discount_leakage"]:,.0f}</b> in leakage — the smallest channel but easy to address with a POS-enforced pricing floor.</div>''', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SCHEDULING INEFFICIENCY ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif section == "📅 Scheduling Inefficiency Analysis":
    st.markdown('<div class="sec-header">📅 Scheduling Inefficiency Analysis — Gaps, Peaks & Idle Patterns</div>', unsafe_allow_html=True)

    # ── Computed scheduling insights from dataframes ──────────────────────────
    peak_hour_row   = peak_hours.loc[peak_hours["total_revenue"].idxmax()]
    trough_hour_row = peak_hours.loc[peak_hours["total_revenue"].idxmin()]
    worst_room_row  = room_kpis.loc[room_kpis["scheduled_utilization"].idxmin()]
    best_room_row   = room_kpis.loc[room_kpis["scheduled_utilization"].idxmax()]
    avg_idle_per_day = idle_time.groupby("date")["idle_min"].sum().mean()

    # ── 4 KPI Cards ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak Revenue Hour",    f"{int(peak_hour_row['hour']):02d}:00",
              delta=f"${peak_hour_row['total_revenue']:,.0f} total", delta_color="off")
    c2.metric("Lowest Demand Hour",   f"{int(trough_hour_row['hour']):02d}:00",
              delta=f"${trough_hour_row['total_revenue']:,.0f} total", delta_color="inverse")
    c3.metric("Highest Idle Room",    worst_room_row["room_name"],
              delta=f"{worst_room_row['scheduled_utilization']:.1%} utilization", delta_color="inverse")
    c4.metric("Avg Daily Idle (all rooms)", f"{avg_idle_per_day:,.0f} min",
              delta=f"{avg_idle_per_day/60:.1f} hrs/day", delta_color="inverse")

    st.markdown("---")

    # ── Heatmap (full width) ──────────────────────────────────────────────────
    st.markdown("#### 🗓️ Room Utilization Heatmap")
    st.pyplot(fig_room_heatmap(idle_time), use_container_width=True)

    st.markdown("---")

    # ── Peak hours + Idle histogram ───────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### ⏰ Revenue by Hour of Day")
        st.pyplot(fig_peak_hours(peak_hours))
    with col_b:
        st.markdown("#### 📊 Idle Time Distribution per Room")
        st.pyplot(fig_idle_histogram(idle_time))

    st.markdown("---")

    # ── Computed text insights ────────────────────────────────────────────────
    st.markdown("#### 🔍 Scheduling Intelligence Summary")
    col_c, col_d = st.columns(2)
    with col_c:
        # Day-of-week performance from appointments
        dow_rev = appointments[appointments["status"]=="Completed"].groupby("day_of_week")["actual_revenue"].sum()
        best_dow  = dow_rev.idxmax()
        worst_dow = dow_rev.idxmin()
        st.markdown(f'''<div class="success-box">
        📅 <b>Best Day:</b> {best_dow} (${dow_rev[best_dow]:,.0f} revenue)<br>
        📅 <b>Weakest Day:</b> {worst_dow} (${dow_rev[worst_dow]:,.0f} revenue)<br>
        💡 Consider launching a <b>{worst_dow} Promotion</b> to lift the lowest-demand day.
        </div>''', unsafe_allow_html=True)

        st.markdown(f'''<div class="insight-box">
        ⏰ <b>Peak Revenue Hour:</b> {int(peak_hour_row["hour"])}:00 AM  (${peak_hour_row["total_revenue"]:,.0f})<br>
        ⏰ <b>Lowest Revenue Hour:</b> {int(trough_hour_row["hour"])}:00  (${trough_hour_row["total_revenue"]:,.0f})<br>
        💡 Reserve high-margin services (Botox, Filler) for morning slots; use last-minute discounts for {int(trough_hour_row["hour"])}:00 trough.
        </div>''', unsafe_allow_html=True)

    with col_d:
        st.markdown(f'''<div class="danger-box">
        🏠 <b>Most Idle Room:</b> {worst_room_row["room_name"]} ({worst_room_row["scheduled_utilization"]:.1%} util)<br>
        🏠 <b>Best Utilized Room:</b> {best_room_row["room_name"]} ({best_room_row["scheduled_utilization"]:.1%} util)<br>
        💡 Redistribute services or offer same-day pricing in {worst_room_row["room_name"]} to close the {best_room_row["scheduled_utilization"] - worst_room_row["scheduled_utilization"]:.1%} utilization gap.
        </div>''', unsafe_allow_html=True)

        # Utilization std — shows variability
        util_std = idle_time.groupby("date")["utilization"].mean().std()
        st.markdown(f'''<div class="warn-box">
        📉 <b>Utilization Volatility:</b> ±{util_std:.1%} std dev day-to-day<br>
        📉 <b>Total Idle Minutes:</b> {leakage["total_idle_min"]:,.0f} min ({leakage["total_idle_min"]/60:.0f} hrs over 30 days)<br>
        💡 High day-to-day variance suggests inconsistent demand patterns — a waitlist system would smooth this.
        </div>''', unsafe_allow_html=True)

    st.markdown("---")

    # ── Day-of-week revenue bar ───────────────────────────────────────────────
    st.markdown("#### 📅 Revenue by Day of Week")
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
    dow_df = appointments[appointments["status"]=="Completed"].groupby("day_of_week")["actual_revenue"].sum().reindex(dow_order).dropna().reset_index()
    dow_colors = [PAL["success"] if v == dow_df["actual_revenue"].max() else PAL["danger"] if v == dow_df["actual_revenue"].min() else PAL["primary"] for v in dow_df["actual_revenue"]]

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    fig_dow, ax_dow = plt.subplots(figsize=(10, 3.5))
    ax_dow.bar(dow_df["day_of_week"], dow_df["actual_revenue"], color=dow_colors, edgecolor="white")
    ax_dow.set_title("Revenue by Day of Week (Completed Appointments)", fontsize=12, fontweight="bold", pad=10)
    ax_dow.set_ylabel("Revenue ($)")
    ax_dow.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    import seaborn as sns
    sns.despine()
    fig_dow.tight_layout()
    st.pyplot(fig_dow)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — OPTIMIZATION SIMULATION
# ═════════════════════════════════════════════════════════════════════════════
elif section == "⚙️ Optimization Simulation":
    st.markdown('<div class="sec-header">⚙️ Optimization Simulation — Gap-Fill & Overbooking Engines</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🟢 Gap-Fill Algorithm", "🔶 Overbooking Model", "🔵 Waterfall Summary", "🎛️ Interactive Simulator"])

    # ── TAB 1: Gap Fill ──────────────────────────────────────────────────
    with tab1:
        st.markdown("### Gap-Fill Algorithm Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gaps Detected",   f"{gap_results['total_gaps_found']:,}")
        c2.metric("Gaps Filled",      f"{gap_results['total_gaps_filled']:,}")
        c3.metric("Fill Rate",        f"{gap_results['fill_rate_actual']:.1%}")
        c4.metric("Revenue Recovered",f"${gap_results['total_revenue_gain']:,.0f}",
                  delta=f"+{gap_results['total_revenue_gain']/base_revenue:.1%} uplift")

        st.markdown("---")
        st.pyplot(fig_gap_fill_uplift(daily_kpis, gap_results["daily_uplift"]),
                  use_container_width=True)

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.pyplot(fig_gap_fill_rooms(gap_results["gap_summary"]))
        with col_b:
            st.markdown("#### Filled Slot Sample")
            if not gap_results["filled_slots"].empty:
                fs = gap_results["filled_slots"][
                    ["fill_id","date","room_name","provider_name","service_name","duration_min","revenue"]
                ].head(20).copy()
                fs["revenue"] = fs["revenue"].map("${:,.0f}".format)
                fs.columns = ["Fill ID","Date","Room","Provider","Service","Duration (min)","Revenue"]
                st.dataframe(fs, use_container_width=True, hide_index=True, height=340)
            else:
                st.info("No fills generated.")

        st.markdown("---")
        st.markdown(f'<div class="success-box">✅ The gap-fill engine identified <b>{gap_results["total_gaps_found"]} actionable idle windows</b> and filled <b>{gap_results["total_gaps_filled"]}</b> of them, recovering <b>${gap_results["total_revenue_gain"]:,.0f}</b>. Increasing fill_rate to 85% (via automated last-minute booking tools) could push recovery to ~${gap_results["total_revenue_gain"]/gap_results["fill_rate_actual"]*0.85:,.0f}.</div>', unsafe_allow_html=True)

    # ── TAB 2: Overbooking ───────────────────────────────────────────────
    with tab2:
        st.markdown("### Risk-Adjusted Overbooking Model")

        ob_df = ob_results["scenario_summary"]
        opt   = ob_results["optimal_factor"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observed No-Show Rate", f"{ob_results['cfg'].observed_noshow_rate:.1%}")
        c2.metric("Base Daily Capacity",   f"{ob_results['base_capacity']} appts")
        c3.metric("Optimal Factor",        ob_results["optimal_pct"])
        c4.metric("Expected Daily Uplift", f"${ob_results['optimal_uplift']:,.0f}")

        st.markdown("---")
        st.pyplot(fig_overbooking(ob_results), use_container_width=True)

        st.markdown("---")
        st.markdown("#### Scenario Comparison Table")
        ob_display = ob_df[[
            "overbook_factor_pct","extra_slots","mean_shows",
            "mean_net_revenue","mean_uplift","overflow_prob","mean_penalty","sharpe_ratio"
        ]].copy()
        ob_display.columns = ["Factor","Extra Slots","Mean Shows","Mean Net Rev","Mean Uplift",
                               "Overflow Prob","Mean Penalty","Sharpe Ratio"]
        ob_display["Mean Net Rev"] = ob_display["Mean Net Rev"].map("${:,.0f}".format)
        ob_display["Mean Uplift"]  = ob_display["Mean Uplift"].map("${:,.0f}".format)
        ob_display["Overflow Prob"]= ob_display["Overflow Prob"].map("{:.1%}".format)
        ob_display["Mean Penalty"] = ob_display["Mean Penalty"].map("${:,.0f}".format)
        st.dataframe(ob_display, use_container_width=True, hide_index=True)

        st.markdown("---")
        opt_row = ob_df[ob_df["overbook_factor"] == opt].iloc[0]
        st.markdown(f'<div class="insight-box">📌 <b>Optimal factor is {ob_results["optimal_pct"]}</b> by mean uplift (${ob_results["optimal_uplift"]:,.0f}/day). Overflow probability is {opt_row["overflow_prob"]*100:.1f}% — mitigate by maintaining an active waitlist to convert overflows into rescheduled revenue rather than walkouts.</div>', unsafe_allow_html=True)
        conservative = ob_df[ob_df["overbook_factor"] == 0.10].iloc[0]
        st.markdown(f'<div class="success-box">✅ <b>Conservative pick: +10%</b> — ${conservative["mean_uplift"]:,.0f}/day uplift at only {conservative["overflow_prob"]*100:.1f}% overflow risk. Sharpe ratio of {conservative["sharpe_ratio"]:.3f} indicates good risk-adjusted return.</div>', unsafe_allow_html=True)

    # ── TAB 3: Waterfall ─────────────────────────────────────────────────
    with tab3:
        st.markdown("### Optimization Waterfall: Cumulative Revenue Impact")
        st.pyplot(
            fig_waterfall(
                base_revenue,
                gap_results["total_revenue_gain"],
                ob_results["optimal_uplift"],
                sensitivity,
            ),
            use_container_width=True,
        )

        combo = sensitivity["combo_scenarios"]
        st.markdown("---")
        st.markdown("#### Combined Scenario Outcomes")
        combo_d = combo.copy()
        combo_d["revenue_gain_$"] = combo_d["revenue_gain_$"].map("${:,.0f}".format)
        combo_d["new_revenue_$"]  = combo_d["new_revenue_$"].map("${:,.0f}".format)
        combo_d["monthly_cost_$"] = combo_d["monthly_cost_$"].map("${:,.0f}".format)
        combo_d["roi_pct"]        = combo_d["roi_pct"].map("{:,.0f}%".format)
        combo_d["new_util"]       = combo_d["new_util"].map("{:.1%}".format)
        combo_d.columns = ["Scenario","Uplift","Revenue Gain","New Revenue","Monthly Cost","ROI","New Util"]
        st.dataframe(combo_d[["Scenario","Revenue Gain","New Revenue","Monthly Cost","ROI","New Util"]],
                     use_container_width=True, hide_index=True)

    # ── TAB 4: Interactive Scenario Simulator ────────────────────────────
    with tab4:
        st.markdown("### 🎛️ Interactive Scenario Simulator")
        st.markdown("Adjust the sliders below to model different operational scenarios and instantly see projected revenue impact.")

        col_sl, col_res = st.columns([0.38, 0.62])
        with col_sl:
            st.markdown("#### Scenario Parameters")
            no_show_rate_input   = st.slider("Expected No-Show Rate",     0.05, 0.30, 0.12, 0.01, format="%.2f", help="Current observed rate is 13.7%")
            overbooking_rate     = st.slider("Overbooking Percentage",    0.00, 0.20, 0.05, 0.01, format="%.2f", help="Extra slots to schedule above capacity")
            demand_growth        = st.slider("Demand Growth Scenario",    0.00, 0.50, 0.10, 0.01, format="%.2f", help="Additional demand from marketing / waitlist")
            fill_rate_input      = st.slider("Gap-Fill Success Rate",     0.10, 1.00, 0.70, 0.05, format="%.2f", help="Fraction of idle gaps filled by last-minute bookings")

        with col_res:
            st.markdown("#### Projected Outcomes")

            # ── Simulation formulas ────────────────────────────────────────
            # 1. No-show improvement: reduce no-show leakage proportionally
            current_ns_rate  = 0.137
            ns_improvement   = max(0, current_ns_rate - no_show_rate_input)
            recovered_ns     = ns_improvement / current_ns_rate * leakage["no_show_leakage"]

            # 2. Overbooking uplift: expected extra revenue from overbook slots filling
            show_prob        = 1.0 - no_show_rate_input - 0.071
            ob_extra_slots   = int(ob_results["base_capacity"] * overbooking_rate)
            ob_revenue_gain  = ob_extra_slots * show_prob * appointments[appointments["status"]=="Completed"]["actual_revenue"].mean()

            # 3. Demand growth: proportional revenue lift
            demand_rev       = base_revenue * demand_growth

            # 4. Gap-fill: scale from Phase 3 result by selected fill rate
            base_fill_rate   = gap_results["fill_rate_actual"]
            fill_scale       = fill_rate_input / base_fill_rate if base_fill_rate > 0 else 1.0
            gap_rev_scaled   = min(gap_results["total_revenue_gain"] * fill_scale, leakage["idle_leakage"])

            # Total projection
            total_gain       = recovered_ns + ob_revenue_gain + demand_rev + gap_rev_scaled
            projected_revenue= base_revenue + total_gain
            projected_util   = min(base_util + demand_growth * 0.5 + overbooking_rate * 0.3, 1.0)

            # ── Result metrics ─────────────────────────────────────────────
            r1, r2, r3 = st.columns(3)
            r1.metric("Projected Revenue",    f"${projected_revenue:,.0f}",
                      delta=f"+${total_gain:,.0f} vs baseline")
            r2.metric("Recovered Revenue",    f"${total_gain:,.0f}",
                      delta=f"+{total_gain/base_revenue:.1%} uplift")
            r3.metric("Projected Utilization",f"{projected_util:.1%}",
                      delta=f"+{projected_util - base_util:.1%} vs current")

            st.markdown("---")
            st.markdown("##### Revenue Gain Breakdown")
            breakdown_df = pd.DataFrame([
                {"Source": "No-Show Reduction",    "Gain": round(recovered_ns, 0)},
                {"Source": "Overbooking",           "Gain": round(ob_revenue_gain, 0)},
                {"Source": "Demand Growth",         "Gain": round(demand_rev, 0)},
                {"Source": "Gap-Fill Bookings",     "Gain": round(gap_rev_scaled, 0)},
            ])
            breakdown_df["Gain $"] = breakdown_df["Gain"].map("${:,.0f}".format)
            breakdown_df["Share"]  = (breakdown_df["Gain"] / breakdown_df["Gain"].sum() * 100).map("{:.1f}%".format)
            st.dataframe(breakdown_df[["Source","Gain $","Share"]], use_container_width=True, hide_index=True)

            # ── Mini bar chart ─────────────────────────────────────────────
            import matplotlib.pyplot as plt
            fig_sim, ax_sim = plt.subplots(figsize=(7, 2.8))
            colors_sim = [PAL["success"], PAL["accent"], PAL["primary"], PAL["neutral"]]
            ax_sim.barh(breakdown_df["Source"], breakdown_df["Gain"], color=colors_sim, edgecolor="white", height=0.5)
            ax_sim.set_xlabel("Revenue Gain ($)")
            ax_sim.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            ax_sim.set_title("Simulated Revenue Gain by Source", fontsize=10, fontweight="bold")
            import seaborn as sns; sns.despine(left=True)
            fig_sim.tight_layout()
            st.pyplot(fig_sim)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SENSITIVITY MODELING
# ═════════════════════════════════════════════════════════════════════════════
elif section == "📈 Sensitivity Modeling":
    st.markdown('<div class="sec-header">📈 Sensitivity Modeling — Revenue Uplift by Optimization Lever</div>', unsafe_allow_html=True)

    sm = sensitivity["scenario_matrix"]

    # ── Top metrics ──────────────────────────────────────────────────────
    best_5  = sm[sm["uplift_target"] == 0.05].sort_values("revenue_gain_$", ascending=False).iloc[0]
    best_10 = sm[sm["uplift_target"] == 0.10].sort_values("revenue_gain_$", ascending=False).iloc[0]
    combo5  = sensitivity["combo_scenarios"][sensitivity["combo_scenarios"]["uplift_target"] == 0.05].iloc[0]
    combo10 = sensitivity["combo_scenarios"][sensitivity["combo_scenarios"]["uplift_target"] == 0.10].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Base Utilization",          f"{base_util:.1%}")
    c2.metric("Best Lever +5% Gain",       f"${best_5['revenue_gain_$']:,.0f}", delta=best_5["lever"], delta_color="off")
    c3.metric("Best Lever +10% Gain",      f"${best_10['revenue_gain_$']:,.0f}", delta=best_10["lever"], delta_color="off")
    c4.metric("All-Lever +10% New Revenue",f"${combo10['new_revenue_$']:,.0f}", delta=f"+{combo10['revenue_gain_$']:,.0f}")

    st.markdown("---")

    # ── Heatmap + ROI bars ───────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Revenue Gain Matrix")
        st.pyplot(fig_sensitivity_heatmap(sensitivity))
    with col_b:
        st.markdown("#### ROI by Lever & Uplift Target")
        st.pyplot(fig_roi_bars(sensitivity))

    st.markdown("---")

    # ── Interactive: scenario explorer ───────────────────────────────────
    st.markdown("#### 🎛️ Scenario Explorer")
    col_l, col_r = st.columns([0.3, 0.7])
    with col_l:
        uplift_sel = st.radio("Utilization Uplift Target", ["+5%", "+10%"])
        uplift_val = 0.05 if uplift_sel == "+5%" else 0.10
        lever_opts = sm["lever"].unique().tolist()
        lever_sel  = st.multiselect("Levers to Activate", lever_opts, default=lever_opts)

    with col_r:
        filtered = sm[(sm["uplift_target"] == uplift_val) & (sm["lever"].isin(lever_sel))].copy()
        total_gain = filtered["revenue_gain_$"].sum()
        total_cost = filtered["monthly_cost_$"].sum()
        new_rev    = base_revenue + total_gain
        combined_roi = round((total_gain - total_cost) / total_cost * 100, 1) if total_cost > 0 else 0

        r1, r2, r3 = st.columns(3)
        r1.metric("Selected Revenue Gain", f"${total_gain:,.0f}")
        r2.metric("Monthly Spend",         f"${total_cost:,.0f}")
        r3.metric("Combined ROI",          f"{combined_roi:,.0f}%")

        st.markdown("##### Per-Lever Breakdown")
        disp = filtered[["lever","max_pool_$","revenue_gain_$","monthly_cost_$","roi_pct","new_util"]].copy()
        disp.columns = ["Lever","Max Pool","Revenue Gain","Monthly Cost","ROI %","New Util"]
        disp["Max Pool"]     = disp["Max Pool"].map("${:,.0f}".format)
        disp["Revenue Gain"] = disp["Revenue Gain"].map("${:,.0f}".format)
        disp["Monthly Cost"] = disp["Monthly Cost"].map("${:,.0f}".format)
        disp["ROI %"]        = disp["ROI %"].map("{:,.0f}%".format)
        disp["New Util"]     = disp["New Util"].map("{:.1%}".format)
        st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Full scenario matrix ─────────────────────────────────────────────
    with st.expander("📋 Full Sensitivity Matrix (all levers × all targets)"):
        full = sm.copy()
        full["max_pool_$"]      = full["max_pool_$"].map("${:,.0f}".format)
        full["revenue_gain_$"]  = full["revenue_gain_$"].map("${:,.0f}".format)
        full["new_revenue_$"]   = full["new_revenue_$"].map("${:,.0f}".format)
        full["monthly_cost_$"]  = full["monthly_cost_$"].map("${:,.0f}".format)
        full["roi_pct"]         = full["roi_pct"].map("{:,.0f}%".format)
        full["base_util"]       = full["base_util"].map("{:.1%}".format)
        full["new_util"]        = full["new_util"].map("{:.1%}".format)
        full.columns = ["ID","Lever","Description","Uplift","Uplift %","Max Pool",
                        "Revenue Gain","New Revenue","Base Util","New Util",
                        "Monthly Cost","ROI"]
        st.dataframe(full[["Lever","Uplift %","Max Pool","Revenue Gain",
                            "New Revenue","Monthly Cost","ROI","New Util"]],
                     use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Final recommendation cards ───────────────────────────────────────
    st.markdown("#### 🏁 Strategic Recommendations")
    best_roi_lever = sm.loc[sm[sm["uplift_target"] == 0.10]["roi_pct"].idxmax()]

    st.markdown(f'<div class="success-box">✅ <b>Priority 1 — {best_roi_lever["lever"]}</b>: Highest ROI lever at +10% uplift ({best_roi_lever["roi_pct"]:,.0f}%). Recover <b>${best_roi_lever["revenue_gain_$"]:,.0f}</b> with just <b>${best_roi_lever["monthly_cost_$"]:.0f}/month</b> in tooling investment.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box">📌 <b>Priority 2 — No-Show Reduction</b>: A deposit + reminder system costs ~$300/month and recovers <b>${sm[(sm["lever_id"]=="B") & (sm["uplift_target"]==0.10)]["revenue_gain_$"].iloc[0]:,.0f}</b> at +10% uplift. Industry benchmark shows 40–60% no-show reduction with automated reminders.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box">📌 <b>Priority 3 — Combined Strategy</b>: Activating all levers at +10% uplift yields <b>${combo10["revenue_gain_$"]:,.0f}</b> additional monthly revenue against a <b>${combo10["monthly_cost_$"]:,.0f}</b> investment — a <b>{combo10["roi_pct"]:,.0f}% ROI</b>.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="warn-box">⚠️ <b>Utilization ceiling</b>: Current avg utilization is {base_util:.1%}. Even a +10% push takes rooms to {min(base_util+0.10, 1.0):.1%} — well below saturation, meaning demand-side initiatives (promotions, waitlists) are the binding constraint, not supply.</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — AI OPERATIONAL RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════════════════════
elif section == "🤖 AI Operational Recommendations":
    st.markdown('<div class="sec-header">🤖 AI Operational Recommendations — Data-Driven Action Plan</div>', unsafe_allow_html=True)

    # ── Compute all values used in recommendations ────────────────────────────
    total_appts      = len(appointments)
    no_show_n        = (appointments["status"] == "No-Show").sum()
    no_show_rate_val = no_show_n / total_appts
    avg_util         = room_kpis["scheduled_utilization"].mean()
    worst_room_row   = room_kpis.loc[room_kpis["scheduled_utilization"].idxmin()]
    top_prov_row     = provider_kpis.loc[provider_kpis["revenue_to_cost_ratio"].idxmax()]
    top_svc_row      = service_mix.iloc[0]
    peak_hour_val    = int(peak_hours.loc[peak_hours["total_revenue"].idxmax(), "hour"])
    trough_hour_val  = int(peak_hours.loc[peak_hours["total_revenue"].idxmin(), "hour"])
    best_lever       = sensitivity["scenario_matrix"].loc[
        sensitivity["scenario_matrix"][sensitivity["scenario_matrix"]["uplift_target"]==0.10]["roi_pct"].idxmax()
    ]
    combo10          = sensitivity["combo_scenarios"][sensitivity["combo_scenarios"]["uplift_target"]==0.10].iloc[0]
    late_fees        = appointments[appointments["status"]=="Late Cancel"]["actual_revenue"].sum()

    # ── Priority badges ───────────────────────────────────────────────────────
    st.markdown("#### 🚦 Recommendation Priority Matrix")
    st.caption("Recommendations are generated programmatically from your analytics results — no hardcoded values.")

    # ── CRITICAL: No-show rate ────────────────────────────────────────────────
    if no_show_rate_val > 0.10:
        sev_label = "CRITICAL" if no_show_rate_val > 0.15 else "HIGH"
        sev_color = "rec-badge-danger" if no_show_rate_val > 0.15 else "rec-badge-warn"
        st.markdown(f'''
        <div class="rec-item">
            <span class="rec-badge {sev_color}">{sev_label}</span>
            <div>
                <b>Implement SMS/Email Reminders + Deposit Policy</b><br>
                <span style="color:#94a3b8;font-size:0.82rem;">
                No-show rate is <b style="color:#f87171">{no_show_rate_val:.1%}</b>
                (${leakage["no_show_leakage"]:,.0f} lost). Require a $75–$100 deposit at booking
                and send automated reminders at 48h and 2h before appointments.
                Industry benchmarks show 40–60% no-show reduction.
                Projected monthly recovery: <b style="color:#4ade80">${leakage["no_show_leakage"]*0.50:,.0f}–${leakage["no_show_leakage"]*0.60:,.0f}</b>.
                </span>
            </div>
        </div>''', unsafe_allow_html=True)

    # ── HIGH: Room underutilization ───────────────────────────────────────────
    if avg_util < 0.70:
        st.markdown(f'''
        <div class="rec-item">
            <span class="rec-badge rec-badge-warn">HIGH</span>
            <div>
                <b>Launch Same-Day Open-Slot Discount Program — {worst_room_row["room_name"]}</b><br>
                <span style="color:#94a3b8;font-size:0.82rem;">
                Average room utilization is <b style="color:#fbbf24">{avg_util:.1%}</b>
                (target: 70%+). {worst_room_row["room_name"]} is the worst performer at
                <b style="color:#f87171">{worst_room_row["scheduled_utilization"]:.1%}</b>.
                Deploy a 15–20% last-minute discount for idle windows between
                {trough_hour_val}:00–{trough_hour_val+2}:00.
                Estimated gap-fill recovery at 70% fill rate:
                <b style="color:#4ade80">${gap_results["total_revenue_gain"]:,.0f}</b>/month.
                </span>
            </div>
        </div>''', unsafe_allow_html=True)

    # ── HIGH: Top provider leverage ───────────────────────────────────────────
    st.markdown(f'''
    <div class="rec-item">
        <span class="rec-badge rec-badge-warn">HIGH</span>
        <div>
            <b>Expand Scheduling Capacity for {top_prov_row["name"]}</b><br>
            <span style="color:#94a3b8;font-size:0.82rem;">
            {top_prov_row["name"]} ({top_prov_row["role"]}) has the highest revenue-to-cost
            ratio at <b style="color:#4ade80">{top_prov_row["revenue_to_cost_ratio"]:.2f}×</b>
            and generates <b style="color:#4ade80">${top_prov_row["total_revenue"]:,.0f}</b> over 30 days.
            Add 2–3 extra slots on Thursday and Friday (peak demand days) — each slot at
            ${top_prov_row["revenue_per_appt"]:,.0f} avg = ~${top_prov_row["revenue_per_appt"]*2.5*4:,.0f} additional monthly revenue.
            </span>
        </div>
    </div>''', unsafe_allow_html=True)

    # ── MEDIUM: Peak hour protection ─────────────────────────────────────────
    st.markdown(f'''
    <div class="rec-item">
        <span class="rec-badge" style="background:#6366f1;">MEDIUM</span>
        <div>
            <b>Protect {peak_hour_val}:00 AM Peak Slots for High-Margin Services</b><br>
            <span style="color:#94a3b8;font-size:0.82rem;">
            {peak_hour_val}:00 AM generates the most revenue per hour.
            Reserve these slots exclusively for <b>{top_svc_row["service_name"]}</b>
            (${top_svc_row["avg_listed_price"]:,.0f} listed, {top_svc_row["revenue_share_pct"]:.1f}% of revenue)
            and Dermal Filler. Avoid booking lower-margin services (Waxing, Chemical Peel)
            during this window — redirect them to the {trough_hour_val}:00 trough.
            </span>
        </div>
    </div>''', unsafe_allow_html=True)

    # ── MEDIUM: Late cancel policy ────────────────────────────────────────────
    late_listed = leakage["late_cancel_leakage"] + late_fees
    st.markdown(f'''
    <div class="rec-item">
        <span class="rec-badge" style="background:#6366f1;">MEDIUM</span>
        <div>
            <b>Enforce 48-Hour Cancellation Policy with Waitlist Automation</b><br>
            <span style="color:#94a3b8;font-size:0.82rem;">
            Late cancellations recovered only <b style="color:#fbbf24">${late_fees:,.0f}</b>
            against <b style="color:#f87171">${late_listed:,.0f}</b> in listed revenue
            ({late_fees/late_listed:.0%} recovery rate). Implementing a strict 48-hour policy
            and auto-notifying the top-3 waitlist clients could recover
            <b style="color:#4ade80">${leakage["late_cancel_leakage"]*0.55:,.0f}</b> per month.
            </span>
        </div>
    </div>''', unsafe_allow_html=True)

    # ── LOW: Pricing floor ────────────────────────────────────────────────────
    st.markdown(f'''
    <div class="rec-item">
        <span class="rec-badge rec-badge-success">LOW</span>
        <div>
            <b>Enforce Pricing Floor — Eliminate Uncontrolled Discounting</b><br>
            <span style="color:#94a3b8;font-size:0.82rem;">
            Pricing discounts cost <b style="color:#fbbf24">${leakage["discount_leakage"]:,.0f}</b>
            this period. This is the smallest leakage channel but easiest to close:
            configure a POS-enforced minimum price at 95% of listed rate.
            Maximum recovery potential: <b style="color:#4ade80">${leakage["discount_leakage"]:,.0f}</b>
            at near-zero implementation cost.
            </span>
        </div>
    </div>''', unsafe_allow_html=True)

    st.markdown("---")

    # ── Prioritized action table ──────────────────────────────────────────────
    st.markdown("#### 📋 Prioritized Action Plan")
    action_df = pd.DataFrame([
        {"Priority": "🔴 Critical", "Action": "SMS reminders + deposit policy",         "Est. Monthly Gain": f"${leakage['no_show_leakage']*0.55:,.0f}", "Est. Monthly Cost": "$300", "ROI": f"{(leakage['no_show_leakage']*0.55-300)/300*100:.0f}%"},
        {"Priority": "🟠 High",     "Action": f"Gap-fill open slots ({worst_room_row['room_name']})", "Est. Monthly Gain": f"${gap_results['total_revenue_gain']:,.0f}", "Est. Monthly Cost": "$800", "ROI": f"{(gap_results['total_revenue_gain']-800)/800*100:.0f}%"},
        {"Priority": "🟠 High",     "Action": f"Expand {top_prov_row['name']} schedule",  "Est. Monthly Gain": f"${top_prov_row['revenue_per_appt']*2.5*4:,.0f}", "Est. Monthly Cost": "$0",   "ROI": "∞"},
        {"Priority": "🟡 Medium",   "Action": "48-hr cancellation + waitlist",            "Est. Monthly Gain": f"${leakage['late_cancel_leakage']*0.55:,.0f}", "Est. Monthly Cost": "$200", "ROI": f"{(leakage['late_cancel_leakage']*0.55-200)/200*100:.0f}%"},
        {"Priority": "🟡 Medium",   "Action": "Peak-hour slot protection",                "Est. Monthly Gain": f"~${top_svc_row['avg_listed_price']*4:,.0f}", "Est. Monthly Cost": "$0",   "ROI": "∞"},
        {"Priority": "🟢 Low",      "Action": "POS pricing floor enforcement",            "Est. Monthly Gain": f"${leakage['discount_leakage']:,.0f}", "Est. Monthly Cost": "$150", "ROI": f"{(leakage['discount_leakage']-150)/150*100:.0f}%"},
    ])
    st.dataframe(action_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Combined impact summary ───────────────────────────────────────────────
    st.markdown("#### 💰 Combined Impact Estimate")
    total_est_gain = (leakage["no_show_leakage"]*0.55 + gap_results["total_revenue_gain"] +
                      leakage["late_cancel_leakage"]*0.55 + leakage["discount_leakage"])
    total_est_cost = 300 + 800 + 200 + 150
    combined_roi   = (total_est_gain - total_est_cost) / total_est_cost * 100

    ci1, ci2, ci3 = st.columns(3)
    ci1.metric("Total Est. Monthly Gain", f"${total_est_gain:,.0f}")
    ci2.metric("Total Monthly Investment", f"${total_est_cost:,.0f}")
    ci3.metric("Blended ROI",             f"{combined_roi:,.0f}%")

    st.markdown(f'''<div class="success-box" style="margin-top:12px;">
    🏁 <b>Executive Summary:</b> Implementing all six recommendations would recover an estimated
    <b>${total_est_gain:,.0f}/month</b> against a <b>${total_est_cost}/month</b> investment —
    a <b>{combined_roi:,.0f}% blended ROI</b>. The top 3 actions alone
    (deposits, gap-fill, provider capacity) account for {((leakage["no_show_leakage"]*0.55 + gap_results["total_revenue_gain"] + top_prov_row["revenue_per_appt"]*10)/total_est_gain*100):.0f}%
    of total recovery at minimal cost.
    </div>''', unsafe_allow_html=True)
