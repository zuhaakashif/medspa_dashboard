"""
Phase 2 — Utilization & Revenue Analytics
Med Spa Revenue Leakage & Utilization Intelligence Sandbox

Consumes DataFrames from Phase 1 and produces:
  - KPI DataFrames: room, provider, daily, service-mix
  - Revenue leakage quantification
  - Advanced insights: peak hours, room-type efficiency, provider efficiency
  - Matplotlib / Seaborn visualizations saved to /outputs
  - Executive summary printed to stdout
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime

warnings.filterwarnings("ignore")

# ── bring Phase 1 into scope ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from phase1_data_engine import generate_all_data, BUSINESS_OPEN, BUSINESS_CLOSE

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
AVAILABLE_MIN_PER_DAY = (BUSINESS_CLOSE - BUSINESS_OPEN) * 60  # 600 min
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Consistent colour palette used across all charts
PALETTE = {
    "primary":   "#2D6A9F",
    "accent":    "#E8A838",
    "danger":    "#D94F3D",
    "success":   "#3DAD77",
    "neutral":   "#8E9BAE",
    "bg":        "#F7F9FC",
    "grid":      "#E2E8F0",
}
ROOM_COLORS  = ["#2D6A9F", "#3DAD77", "#E8A838", "#D94F3D"]
PROV_COLORS  = ["#2D6A9F", "#5B8DB8", "#8BBAD4", "#BDD7E9"]

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor":   PALETTE["bg"],
    "axes.edgecolor":   PALETTE["grid"],
    "grid.color":       PALETTE["grid"],
    "font.family":      "DejaVu Sans",
})


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — KPI COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_room_kpis(appointments: pd.DataFrame, idle_time: pd.DataFrame) -> pd.DataFrame:
    """
    Per-room KPIs aggregated over the full 30-day window.

    Columns produced:
      scheduled_utilization  — based on booked (scheduled) minutes
      actual_utilization     — based on minutes actually delivered (excl. no-shows)
      total_revenue          — sum of actual_revenue
      avg_daily_revenue      — mean revenue on days the room operated
      revenue_per_booked_min — revenue density
      total_idle_min         — cumulative idle minutes
      no_show_count          — appointments with status No-Show
      late_cancel_count      — appointments with status Late Cancel
    """
    operating_days = appointments["date"].nunique()

    # Revenue and appointment counts from the appointments table
    rev = (
        appointments
        .groupby("room_id")
        .agg(
            total_revenue        = ("actual_revenue",          "sum"),
            total_appts          = ("appointment_id",          "count"),
            completed_appts      = ("status",   lambda s: (s == "Completed").sum()),
            no_show_count        = ("status",   lambda s: (s == "No-Show").sum()),
            late_cancel_count    = ("status",   lambda s: (s == "Late Cancel").sum()),
            booked_min_total     = ("scheduled_duration_min",  "sum"),
            actual_min_total     = ("actual_duration_min",     "sum"),
        )
        .reset_index()
    )

    # Idle time totals from the idle_time table (already per room/day)
    idle_agg = (
        idle_time
        .groupby("room_id")
        .agg(total_idle_min = ("idle_min", "sum"))
        .reset_index()
    )

    df = rev.merge(idle_agg, on="room_id")

    # Total available minutes over the simulation
    total_avail = operating_days * AVAILABLE_MIN_PER_DAY

    df["scheduled_utilization"]  = (df["booked_min_total"]  / total_avail).round(4)
    df["actual_utilization"]     = (df["actual_min_total"]  / total_avail).round(4)
    df["avg_daily_revenue"]      = (df["total_revenue"]      / operating_days).round(2)
    df["revenue_per_booked_min"] = np.where(
        df["booked_min_total"] > 0,
        (df["total_revenue"] / df["booked_min_total"]).round(4),
        0,
    )

    # Merge room metadata for context
    from phase1_data_engine import ROOMS
    df = df.merge(ROOMS[["room_id", "room_name", "room_type"]], on="room_id")

    return df[[
        "room_id", "room_name", "room_type",
        "scheduled_utilization", "actual_utilization",
        "total_revenue", "avg_daily_revenue", "revenue_per_booked_min",
        "total_idle_min", "no_show_count", "late_cancel_count",
        "total_appts", "completed_appts",
    ]]


def compute_provider_kpis(appointments: pd.DataFrame) -> pd.DataFrame:
    """
    Per-provider KPIs aggregated over the full 30-day window.
    """
    from phase1_data_engine import PROVIDERS

    operating_days = appointments["date"].nunique()
    total_avail    = operating_days * AVAILABLE_MIN_PER_DAY

    df = (
        appointments
        .groupby("provider_id")
        .agg(
            total_revenue       = ("actual_revenue",         "sum"),
            total_appts         = ("appointment_id",         "count"),
            completed_appts     = ("status", lambda s: (s == "Completed").sum()),
            no_show_count       = ("status", lambda s: (s == "No-Show").sum()),
            late_cancel_count   = ("status", lambda s: (s == "Late Cancel").sum()),
            booked_min_total    = ("scheduled_duration_min", "sum"),
            actual_min_total    = ("actual_duration_min",    "sum"),
        )
        .reset_index()
    )

    df["scheduled_utilization"] = (df["booked_min_total"] / total_avail).round(4)
    df["actual_utilization"]    = (df["actual_min_total"] / total_avail).round(4)
    df["avg_daily_revenue"]     = (df["total_revenue"]    / operating_days).round(2)
    df["revenue_per_appt"]      = np.where(
        df["completed_appts"] > 0,
        (df["total_revenue"] / df["completed_appts"]).round(2),
        0,
    )

    df = df.merge(PROVIDERS[["provider_id", "name", "role", "hourly_rate"]], on="provider_id")
    df["revenue_to_cost_ratio"] = (
        df["total_revenue"] / (df["hourly_rate"] * df["booked_min_total"] / 60)
    ).round(2)

    return df[[
        "provider_id", "name", "role",
        "scheduled_utilization", "actual_utilization",
        "total_revenue", "avg_daily_revenue", "revenue_per_appt",
        "revenue_to_cost_ratio",
        "no_show_count", "late_cancel_count",
        "total_appts", "completed_appts",
    ]]


def compute_daily_kpis(appointments: pd.DataFrame, idle_time: pd.DataFrame) -> pd.DataFrame:
    """
    Per-day KPIs: actual vs theoretical max revenue, utilization, leakage.
    """
    n_rooms = 4   # static for this simulation

    # Revenue and appointment stats per day
    daily_rev = (
        appointments
        .groupby("date")
        .agg(
            actual_revenue    = ("actual_revenue",         "sum"),
            total_appts       = ("appointment_id",         "count"),
            completed_appts   = ("status", lambda s: (s == "Completed").sum()),
            no_show_count     = ("status", lambda s: (s == "No-Show").sum()),
            late_cancel_count = ("status", lambda s: (s == "Late Cancel").sum()),
            booked_min        = ("scheduled_duration_min", "sum"),
        )
        .reset_index()
    )

    # Idle minutes per day (summed across rooms)
    daily_idle = (
        idle_time
        .groupby("date")
        .agg(total_idle_min = ("idle_min", "sum"))
        .reset_index()
    )

    # Theoretical max: every room fully booked at average revenue/min
    avg_rev_per_min = (
        appointments[appointments["status"] == "Completed"]["actual_revenue"].sum()
        / appointments[appointments["status"] == "Completed"]["actual_duration_min"].sum()
    )
    theoretical_max = n_rooms * AVAILABLE_MIN_PER_DAY * avg_rev_per_min

    df = daily_rev.merge(daily_idle, on="date")
    df["theoretical_max_revenue"] = round(theoretical_max, 2)
    df["utilization"]             = (df["booked_min"] / (n_rooms * AVAILABLE_MIN_PER_DAY)).round(4)
    df["revenue_gap"]             = (df["theoretical_max_revenue"] - df["actual_revenue"]).round(2)
    df["leakage_pct"]             = (df["revenue_gap"] / df["theoretical_max_revenue"]).round(4)
    df["day_of_week"]             = pd.to_datetime(df["date"]).dt.strftime("%A")

    return df


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — REVENUE LEAKAGE
# ═════════════════════════════════════════════════════════════════════════════

def compute_leakage(appointments: pd.DataFrame, idle_time: pd.DataFrame) -> dict:
    """
    Quantifies all revenue leakage channels:
      1. Idle-time leakage  — revenue that could have been earned in unused room time
      2. No-show leakage    — full listed price of no-showed appointments
      3. Late-cancel leakage— listed price minus cancellation fee collected
      4. Discount leakage   — gap between listed and actual price on completed appts

    Returns a dict with scalar summaries and a leakage_breakdown DataFrame.
    """
    # Average revenue per minute (completed only — clean signal)
    completed = appointments[appointments["status"] == "Completed"]
    avg_rev_per_min = (
        completed["actual_revenue"].sum() / completed["actual_duration_min"].sum()
        if completed["actual_duration_min"].sum() > 0 else 0
    )

    # 1. Idle-time leakage
    total_idle_min   = idle_time["idle_min"].sum()
    idle_leakage     = round(total_idle_min * avg_rev_per_min, 2)

    # 2. No-show leakage  (full price lost — no revenue collected)
    no_shows         = appointments[appointments["status"] == "No-Show"]
    no_show_leakage  = round(no_shows["listed_price"].sum(), 2)

    # 3. Late-cancel leakage (listed price − cancellation fee already collected)
    late_cancels          = appointments[appointments["status"] == "Late Cancel"]
    late_cancel_leakage   = round(
        late_cancels["listed_price"].sum() - late_cancels["actual_revenue"].sum(), 2
    )

    # 4. Discount leakage on completed appointments
    discount_leakage = round(
        (completed["listed_price"] - completed["actual_revenue"]).clip(lower=0).sum(), 2
    )

    total_leakage     = idle_leakage + no_show_leakage + late_cancel_leakage + discount_leakage
    theoretical_max   = appointments["listed_price"].sum()
    leakage_pct       = round(total_leakage / theoretical_max * 100, 2) if theoretical_max else 0

    # Per-room idle leakage
    from phase1_data_engine import ROOMS
    room_idle = (
        idle_time
        .groupby("room_id")["idle_min"]
        .sum()
        .reset_index()
        .rename(columns={"idle_min": "idle_min_total"})
    )
    room_idle["idle_leakage_$"] = (room_idle["idle_min_total"] * avg_rev_per_min).round(2)
    room_idle = room_idle.merge(ROOMS[["room_id", "room_name", "room_type"]], on="room_id")

    # Summary breakdown DataFrame
    breakdown = pd.DataFrame([
        {"leakage_type": "Idle Time",           "amount": idle_leakage,          "pct_of_total_leakage": round(idle_leakage / total_leakage * 100, 1)},
        {"leakage_type": "No-Shows",            "amount": no_show_leakage,       "pct_of_total_leakage": round(no_show_leakage / total_leakage * 100, 1)},
        {"leakage_type": "Late Cancellations",  "amount": late_cancel_leakage,   "pct_of_total_leakage": round(late_cancel_leakage / total_leakage * 100, 1)},
        {"leakage_type": "Pricing Discounts",   "amount": discount_leakage,      "pct_of_total_leakage": round(discount_leakage / total_leakage * 100, 1)},
    ])

    return {
        "total_leakage":        total_leakage,
        "leakage_pct":          leakage_pct,
        "idle_leakage":         idle_leakage,
        "no_show_leakage":      no_show_leakage,
        "late_cancel_leakage":  late_cancel_leakage,
        "discount_leakage":     discount_leakage,
        "avg_rev_per_min":      round(avg_rev_per_min, 4),
        "total_idle_min":       total_idle_min,
        "breakdown":            breakdown,
        "room_idle_leakage":    room_idle,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ADVANCED INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════

def compute_peak_hours(appointments: pd.DataFrame) -> pd.DataFrame:
    """
    Identify revenue and volume by hour-of-day to surface peak / trough windows.
    """
    df = appointments[appointments["status"] == "Completed"].copy()
    df["hour"] = df["start_time"].dt.hour

    peak = (
        df.groupby("hour")
        .agg(
            appt_count    = ("appointment_id", "count"),
            total_revenue = ("actual_revenue",  "sum"),
        )
        .reset_index()
    )
    peak["avg_revenue_per_appt"] = (peak["total_revenue"] / peak["appt_count"]).round(2)
    peak["revenue_pct"]          = (peak["total_revenue"] / peak["total_revenue"].sum() * 100).round(2)
    return peak


def compute_service_mix(appointments: pd.DataFrame) -> pd.DataFrame:
    """
    Revenue contribution and volume breakdown by service.
    """
    df = (
        appointments
        .groupby(["service_id", "service_name"])
        .agg(
            total_revenue   = ("actual_revenue",         "sum"),
            appt_count      = ("appointment_id",         "count"),
            completed_count = ("status", lambda s: (s == "Completed").sum()),
            no_show_count   = ("status", lambda s: (s == "No-Show").sum()),
            avg_listed_price= ("listed_price", "mean"),
        )
        .reset_index()
    )
    df["revenue_share_pct"]  = (df["total_revenue"] / df["total_revenue"].sum() * 100).round(2)
    df["completion_rate"]    = (df["completed_count"] / df["appt_count"]).round(4)
    df["avg_actual_revenue"] = (df["total_revenue"] / df["completed_count"].clip(lower=1)).round(2)
    return df.sort_values("total_revenue", ascending=False).reset_index(drop=True)


def compute_room_type_efficiency(appointments: pd.DataFrame, idle_time: pd.DataFrame) -> pd.DataFrame:
    """
    Revenue efficiency grouped by room type — useful for investment decisions.
    """
    from phase1_data_engine import ROOMS
    operating_days = appointments["date"].nunique()

    appt_agg = (
        appointments
        .groupby("room_type")
        .agg(
            total_revenue    = ("actual_revenue",         "sum"),
            total_appts      = ("appointment_id",         "count"),
            booked_min       = ("scheduled_duration_min", "sum"),
        )
        .reset_index()
    )

    # Number of rooms per type
    room_counts = ROOMS.groupby("room_type").size().reset_index(name="room_count")
    appt_agg    = appt_agg.merge(room_counts, on="room_type")

    total_avail_per_type = appt_agg["room_count"] * operating_days * AVAILABLE_MIN_PER_DAY
    appt_agg["utilization"]           = (appt_agg["booked_min"] / total_avail_per_type).round(4)
    appt_agg["revenue_per_room_day"]  = (appt_agg["total_revenue"] / (appt_agg["room_count"] * operating_days)).round(2)
    appt_agg["revenue_per_booked_min"]= (appt_agg["total_revenue"] / appt_agg["booked_min"].clip(lower=1)).round(4)

    return appt_agg.sort_values("revenue_per_room_day", ascending=False).reset_index(drop=True)


def compute_provider_efficiency(appointments: pd.DataFrame) -> pd.DataFrame:
    """
    Provider revenue efficiency including revenue-to-labour-cost ratio.
    """
    from phase1_data_engine import PROVIDERS
    operating_days = appointments["date"].nunique()

    df = (
        appointments[appointments["status"] == "Completed"]
        .groupby("provider_id")
        .agg(
            total_revenue  = ("actual_revenue",      "sum"),
            appt_count     = ("appointment_id",      "count"),
            total_min      = ("actual_duration_min", "sum"),
        )
        .reset_index()
    )
    df = df.merge(PROVIDERS[["provider_id", "name", "role", "hourly_rate"]], on="provider_id")
    df["labour_cost"]            = (df["hourly_rate"] * df["total_min"] / 60).round(2)
    df["revenue_to_cost_ratio"]  = (df["total_revenue"] / df["labour_cost"].clip(lower=1)).round(2)
    df["avg_revenue_per_appt"]   = (df["total_revenue"] / df["appt_count"].clip(lower=1)).round(2)
    df["revenue_per_hour"]       = (df["total_revenue"] / (df["total_min"] / 60).clip(lower=1)).round(2)

    return df.sort_values("revenue_to_cost_ratio", ascending=False).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    return path


def plot_room_utilization_heatmap(idle_time: pd.DataFrame) -> str:
    """
    Heatmap: rooms (y-axis) × date (x-axis), colour = utilization %.
    """
    from phase1_data_engine import ROOMS

    pivot = idle_time.pivot(index="room_id", columns="date", values="utilization")
    # Replace room_id labels with friendly names
    room_labels = ROOMS.set_index("room_id")["room_name"]
    pivot.index  = [room_labels[r] for r in pivot.index]
    pivot.columns = pd.to_datetime(pivot.columns).strftime("%b %d")

    fig, ax = plt.subplots(figsize=(18, 4))
    sns.heatmap(
        pivot * 100,
        ax=ax,
        cmap="YlOrRd",
        vmin=0, vmax=100,
        linewidths=0.4,
        linecolor="#ccc",
        annot=False,
        cbar_kws={"label": "Utilization %", "shrink": 0.8},
    )
    ax.set_title("Room Utilization Heatmap — 30-Day Window", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    fig.tight_layout()
    return _save(fig, "chart_01_room_utilization_heatmap.png")


def plot_revenue_per_provider(provider_kpis: pd.DataFrame) -> str:
    """
    Horizontal bar chart: total revenue per provider with revenue-to-cost ratio annotated.
    """
    df = provider_kpis.sort_values("total_revenue")

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(
        df["name"], df["total_revenue"],
        color=ROOM_COLORS[::-1], edgecolor="white", height=0.55
    )

    # Annotate bars with revenue value and ratio
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(
            bar.get_width() + 500, bar.get_y() + bar.get_height() / 2,
            f"${row['total_revenue']:,.0f}  |  ratio {row['revenue_to_cost_ratio']:.1f}×",
            va="center", fontsize=8.5, color="#333"
        )

    ax.set_title("Total Revenue per Provider (30 days)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Revenue ($)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlim(0, df["total_revenue"].max() * 1.35)
    sns.despine(left=True, bottom=False)
    fig.tight_layout()
    return _save(fig, "chart_02_revenue_per_provider.png")


def plot_daily_revenue_vs_potential(daily_kpis: pd.DataFrame) -> str:
    """
    Dual-line chart: actual daily revenue vs theoretical maximum.
    Shaded area = leakage gap.
    """
    df = daily_kpis.sort_values("date").copy()
    dates = pd.to_datetime(df["date"])

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.fill_between(dates, df["actual_revenue"], df["theoretical_max_revenue"],
                    alpha=0.18, color=PALETTE["danger"], label="Revenue Gap (Leakage)")
    ax.plot(dates, df["theoretical_max_revenue"],
            color=PALETTE["neutral"], linewidth=1.8, linestyle="--", label="Theoretical Max Revenue")
    ax.plot(dates, df["actual_revenue"],
            color=PALETTE["primary"], linewidth=2.2, label="Actual Revenue")

    ax.set_title("Daily Revenue: Actual vs Theoretical Maximum", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    sns.despine()
    fig.tight_layout()
    return _save(fig, "chart_03_daily_revenue_vs_potential.png")


def plot_idle_time_histogram(idle_time: pd.DataFrame) -> str:
    """
    Faceted histogram of idle minutes per room across all operating days.
    """
    from phase1_data_engine import ROOMS
    room_labels = ROOMS.set_index("room_id")["room_name"].to_dict()
    df = idle_time.copy()
    df["room_name"] = df["room_id"].map(room_labels)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, (room_id, color) in zip(axes, zip(ROOMS["room_id"], ROOM_COLORS)):
        sub = df[df["room_id"] == room_id]["idle_min"]
        ax.hist(sub, bins=15, color=color, edgecolor="white", alpha=0.88)
        ax.set_title(room_labels[room_id], fontsize=10, fontweight="bold")
        ax.set_xlabel("Idle Min / Day", fontsize=9)
        ax.axvline(sub.mean(), color="#333", linestyle="--", linewidth=1.2,
                   label=f"Mean: {sub.mean():.0f}m")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Days")
    fig.suptitle("Distribution of Daily Idle Time per Room", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "chart_04_idle_time_histogram.png")


def plot_leakage_breakdown(leakage: dict) -> str:
    """
    Donut chart of leakage by category.
    """
    bd     = leakage["breakdown"]
    labels = bd["leakage_type"].tolist()
    sizes  = bd["amount"].tolist()
    colors = [PALETTE["danger"], PALETTE["accent"], PALETTE["primary"], PALETTE["neutral"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors,
        autopct=lambda p: f"{p:.1f}%", startangle=140,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight("bold")

    legend_labels = [f"{l}  ${v:,.0f}" for l, v in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, fontsize=9)
    ax.set_title(
        f"Revenue Leakage Breakdown\nTotal: ${leakage['total_leakage']:,.0f}  ({leakage['leakage_pct']}% of listed revenue)",
        fontsize=13, fontweight="bold", pad=14,
    )
    fig.tight_layout()
    return _save(fig, "chart_05_leakage_breakdown.png")


def plot_peak_hours(peak_hours: pd.DataFrame) -> str:
    """
    Bar chart of revenue by hour of day, highlighting peak and trough windows.
    """
    df = peak_hours.sort_values("hour")

    fig, ax = plt.subplots(figsize=(11, 4))
    bar_colors = [
        PALETTE["success"] if r == df["total_revenue"].max()
        else PALETTE["danger"] if r == df["total_revenue"].min()
        else PALETTE["primary"]
        for r in df["total_revenue"]
    ]
    ax.bar(df["hour"].astype(str) + ":00", df["total_revenue"],
           color=bar_colors, edgecolor="white")

    ax.set_title("Revenue by Hour of Day (Completed Appointments)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Revenue ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    sns.despine()
    fig.tight_layout()
    return _save(fig, "chart_06_peak_hours.png")


def plot_service_mix(service_mix: pd.DataFrame) -> str:
    """
    Stacked horizontal bar showing revenue share per service.
    """
    df = service_mix.sort_values("total_revenue", ascending=True).head(10)

    cmap   = plt.cm.get_cmap("tab10", len(df))
    colors = [cmap(i) for i in range(len(df))]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(df["service_name"], df["total_revenue"], color=colors, edgecolor="white", height=0.6)
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
                f"{row['revenue_share_pct']:.1f}%  |  {row['appt_count']} appts",
                va="center", fontsize=8.5)
    ax.set_title("Service Revenue Contribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Revenue ($)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlim(0, df["total_revenue"].max() * 1.35)
    sns.despine(left=True)
    fig.tight_layout()
    return _save(fig, "chart_07_service_mix.png")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EXECUTIVE SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

def print_executive_summary(
    appointments: pd.DataFrame,
    room_kpis:     pd.DataFrame,
    provider_kpis: pd.DataFrame,
    daily_kpis:    pd.DataFrame,
    leakage:       dict,
    service_mix:   pd.DataFrame,
    peak_hours:    pd.DataFrame,
) -> None:
    SEP = "═" * 62
    sep = "─" * 62

    print(f"\n{SEP}")
    print("  MED SPA ANALYTICS — EXECUTIVE SUMMARY  (Phase 2)")
    print(f"{SEP}")

    # ── Operational Overview ─────────────────────────────────────────────
    operating_days = appointments["date"].nunique()
    total_rev      = appointments["actual_revenue"].sum()
    total_appts    = len(appointments)
    completed      = (appointments["status"] == "Completed").sum()
    no_shows       = (appointments["status"] == "No-Show").sum()
    late_cancels   = (appointments["status"] == "Late Cancel").sum()

    print(f"\n  OPERATIONAL OVERVIEW")
    print(sep)
    print(f"  Operating days          : {operating_days}")
    print(f"  Total appointments      : {total_appts:,}")
    print(f"  Completed               : {completed:,}  ({completed/total_appts:.1%})")
    print(f"  No-Shows                : {no_shows:,}  ({no_shows/total_appts:.1%})")
    print(f"  Late Cancellations      : {late_cancels:,}  ({late_cancels/total_appts:.1%})")
    print(f"  Total Revenue Collected : ${total_rev:,.2f}")
    print(f"  Avg Daily Revenue       : ${total_rev/operating_days:,.2f}")

    # ── Room KPIs ────────────────────────────────────────────────────────
    print(f"\n  ROOM UTILIZATION KPIs")
    print(sep)
    for _, r in room_kpis.iterrows():
        print(f"  {r['room_name']:<20} | Sched: {r['scheduled_utilization']:.1%}"
              f"  Actual: {r['actual_utilization']:.1%}"
              f"  Rev: ${r['total_revenue']:>9,.0f}"
              f"  Idle: {r['total_idle_min']:>5,.0f} min")

    # ── Provider KPIs ────────────────────────────────────────────────────
    print(f"\n  PROVIDER PERFORMANCE KPIs")
    print(sep)
    for _, p in provider_kpis.sort_values("total_revenue", ascending=False).iterrows():
        print(f"  {p['name']:<22} | Sched: {p['scheduled_utilization']:.1%}"
              f"  Rev: ${p['total_revenue']:>9,.0f}"
              f"  Rev/Appt: ${p['revenue_per_appt']:>6,.0f}"
              f"  R/C: {p['revenue_to_cost_ratio']:.2f}×")

    # ── Leakage ──────────────────────────────────────────────────────────
    print(f"\n  REVENUE LEAKAGE ANALYSIS")
    print(sep)
    print(f"  Total Leakage           : ${leakage['total_leakage']:,.2f}  ({leakage['leakage_pct']}% of listed revenue)")
    for _, row in leakage["breakdown"].iterrows():
        print(f"    {row['leakage_type']:<22}  ${row['amount']:>10,.2f}  ({row['pct_of_total_leakage']:.1f}%)")
    print(f"\n  Total Idle Minutes      : {leakage['total_idle_min']:,.0f} min "
          f"({leakage['total_idle_min']/60:.1f} hrs)")
    print(f"  Avg Revenue / Minute    : ${leakage['avg_rev_per_min']:.4f}")

    # ── Peak Hours ───────────────────────────────────────────────────────
    peak_hour   = peak_hours.loc[peak_hours["total_revenue"].idxmax(), "hour"]
    trough_hour = peak_hours.loc[peak_hours["total_revenue"].idxmin(), "hour"]
    print(f"\n  PEAK / TROUGH HOURS")
    print(sep)
    print(f"  Peak revenue hour       : {peak_hour}:00  (${peak_hours['total_revenue'].max():,.0f})")
    print(f"  Trough revenue hour     : {trough_hour}:00  (${peak_hours['total_revenue'].min():,.0f})")

    # ── Top Services ─────────────────────────────────────────────────────
    print(f"\n  TOP 3 SERVICES BY REVENUE")
    print(sep)
    for _, s in service_mix.head(3).iterrows():
        print(f"  {s['service_name']:<25}  ${s['total_revenue']:>9,.0f}  ({s['revenue_share_pct']:.1f}%)"
              f"  Completion: {s['completion_rate']:.1%}")

    # ── Recommendations ──────────────────────────────────────────────────
    worst_room  = room_kpis.loc[room_kpis["scheduled_utilization"].idxmin(), "room_name"]
    worst_util  = room_kpis["scheduled_utilization"].min()
    top_provider= provider_kpis.loc[provider_kpis["revenue_to_cost_ratio"].idxmax(), "name"]

    print(f"\n  ACTIONABLE RECOMMENDATIONS")
    print(sep)
    print(f"  1. REDUCE NO-SHOW LEAKAGE (${leakage['no_show_leakage']:,.0f} lost)")
    print(f"     → Implement SMS/email reminders 24 h and 2 h before appointment.")
    print(f"       Require a credit card deposit at booking (suggest $50–$100).")
    print()
    print(f"  2. FILL IDLE TIME IN {worst_room.upper()} (util: {worst_util:.1%})")
    print(f"     → Introduce a last-minute 'Open Slot' discount (15–20% off) for")
    print(f"       same-day bookings. Target {trough_hour}:00–{trough_hour+2}:00 off-peak window.")
    print()
    print(f"  3. LEVERAGE TOP REVENUE-TO-COST PROVIDER: {top_provider}")
    print(f"     → Extend scheduling capacity for this provider on high-demand days")
    print(f"       (Thu/Fri). Consider adding a second shift or extended hours.")
    print(f"\n{SEP}\n")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def run_phase2() -> dict:
    """
    Orchestrates all Phase 2 computations and visualizations.
    Returns a dict with all KPI DataFrames, leakage metrics, and chart paths.
    """
    # ── Load Phase 1 data ────────────────────────────────────────────────
    data        = generate_all_data()
    appointments = data["appointments"]
    idle_time    = data["idle_time"]

    print("\n📊  Running Phase 2 — Utilization & Revenue Analytics...")

    # ── KPIs ─────────────────────────────────────────────────────────────
    room_kpis     = compute_room_kpis(appointments, idle_time)
    provider_kpis = compute_provider_kpis(appointments)
    daily_kpis    = compute_daily_kpis(appointments, idle_time)

    # ── Leakage ──────────────────────────────────────────────────────────
    leakage = compute_leakage(appointments, idle_time)

    # ── Advanced insights ────────────────────────────────────────────────
    peak_hours     = compute_peak_hours(appointments)
    service_mix    = compute_service_mix(appointments)
    room_type_eff  = compute_room_type_efficiency(appointments, idle_time)
    provider_eff   = compute_provider_efficiency(appointments)

    # ── Visualizations ───────────────────────────────────────────────────
    print("  🎨  Generating charts...")
    chart_paths = {
        "heatmap":        plot_room_utilization_heatmap(idle_time),
        "rev_provider":   plot_revenue_per_provider(provider_kpis),
        "daily_revenue":  plot_daily_revenue_vs_potential(daily_kpis),
        "idle_histogram": plot_idle_time_histogram(idle_time),
        "leakage_donut":  plot_leakage_breakdown(leakage),
        "peak_hours":     plot_peak_hours(peak_hours),
        "service_mix":    plot_service_mix(service_mix),
    }
    for name, path in chart_paths.items():
        print(f"     ✅  {name:<18}  →  {os.path.basename(path)}")

    # ── Executive Summary ────────────────────────────────────────────────
    print_executive_summary(
        appointments, room_kpis, provider_kpis, daily_kpis,
        leakage, service_mix, peak_hours,
    )

    return {
        # DataFrames
        "room_kpis":       room_kpis,
        "provider_kpis":   provider_kpis,
        "daily_kpis":      daily_kpis,
        "service_mix":     service_mix,
        "peak_hours":      peak_hours,
        "room_type_eff":   room_type_eff,
        "provider_eff":    provider_eff,
        "leakage":         leakage,
        # Chart file paths
        "charts":          chart_paths,
    }


if __name__ == "__main__":
    results = run_phase2()
