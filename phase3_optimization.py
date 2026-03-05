"""
Phase 3 — Optimization Simulation
Med Spa Revenue Leakage & Utilization Intelligence Sandbox

Three simulation engines:
  1. Gap-Filling Algorithm       — inject appointments into idle slots
  2. Risk-Adjusted Overbooking   — overbook by calibrated factor given no-show history
  3. Revenue Sensitivity Analysis— model +5% / +10% utilization uplift scenarios

All engines output DataFrames and charts compatible with the Phase 4 dashboard.
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
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import timedelta
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from phase1_data_engine import (
    generate_all_data, SERVICES, ROOMS, PROVIDERS,
    BUSINESS_OPEN, BUSINESS_CLOSE,
)
from phase2_data_analytics import (
    run_phase2, compute_leakage,
    AVAILABLE_MIN_PER_DAY, OUTPUT_DIR, PALETTE,
    ROOM_COLORS,
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLING  (same palette / theme as Phase 2 for visual consistency)
# ─────────────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor":   PALETTE["bg"],
    "axes.edgecolor":   PALETTE["grid"],
    "grid.color":       PALETTE["grid"],
    "font.family":      "DejaVu Sans",
})

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    return path


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — GAP-FILLING ALGORITHM
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class GapFillConfig:
    """Tuneable parameters for the gap-fill engine."""
    min_gap_threshold_min: int   = 30    # ignore gaps shorter than this
    turnover_buffer_min:   int   = 10    # buffer between injected and existing appts
    fill_rate:             float = 0.70  # fraction of eligible gaps we successfully fill
                                         # (models imperfect last-minute demand)
    avg_revenue_per_min:   float = 0.0   # set at runtime from Phase 2 leakage data


def simulate_gap_filling(
    appointments: pd.DataFrame,
    idle_time:    pd.DataFrame,
    cfg:          GapFillConfig,
) -> dict:
    """
    For every (date, room) pair scan the schedule timeline and identify contiguous
    idle windows large enough to host at least one service.  Inject synthetic
    fill appointments, respecting:
      - Minimum gap threshold
      - Turnover buffer around existing bookings
      - fill_rate (stochastic demand — not every gap fills)
      - Business-hours boundary

    Returns
    -------
    dict with keys:
      filled_slots        DataFrame of injected appointments
      gap_summary         DataFrame per (date, room) with gap/fill metrics
      daily_uplift        DataFrame of revenue uplift per day
      total_revenue_gain  scalar
      total_gaps_found    scalar
      total_gaps_filled   scalar
    """
    import random
    random.seed(99)

    n_rooms   = len(ROOMS)
    day_open  = BUSINESS_OPEN  * 60   # minutes from midnight
    day_close = BUSINESS_CLOSE * 60

    # Build a service lookup by room_type for fast access
    # (only completed-style appointments go in gaps — no no-show injection)
    svc_by_room_type: dict[str, pd.DataFrame] = {}
    for rt in ROOMS["room_type"].unique():
        svc_by_room_type[rt] = SERVICES[
            SERVICES["required_room_type"].apply(lambda x: rt in x)
        ]

    # Provider busy registry (per day) — prevent double-booking injected providers
    # We rebuild it from the base schedule each day
    def _get_provider_busy(day_appts: pd.DataFrame) -> dict[str, list]:
        busy: dict[str, list] = {}
        for _, row in day_appts.iterrows():
            pid = row["provider_id"]
            busy.setdefault(pid, []).append(
                (row["start_time"], row["end_time"])
            )
        return busy

    filled_records = []
    gap_records    = []
    fill_id        = 1

    for date, day_appts in appointments.groupby("date"):
        provider_busy = _get_provider_busy(day_appts)

        for _, room in ROOMS.iterrows():
            room_appts = (
                day_appts[day_appts["room_id"] == room["room_id"]]
                .sort_values("start_time")
                .reset_index(drop=True)
            )

            # Build timeline: list of (start_min, end_min) booked blocks
            booked_blocks = []
            base_dt = pd.Timestamp(str(date))
            for _, appt in room_appts.iterrows():
                s = int((appt["start_time"] - base_dt).total_seconds() // 60)
                e = int((appt["end_time"]   - base_dt).total_seconds() // 60)
                booked_blocks.append((s, e))
            booked_blocks.sort()

            # Add sentinel blocks at open/close to simplify gap detection
            timeline = [(day_open, day_open)] + booked_blocks + [(day_close, day_close)]

            # Scan consecutive pairs for gaps
            for i in range(len(timeline) - 1):
                gap_start = timeline[i][1] + cfg.turnover_buffer_min   # after previous end + buffer
                gap_end   = timeline[i + 1][0] - cfg.turnover_buffer_min  # before next start - buffer
                gap_min   = gap_end - gap_start

                if gap_min < cfg.min_gap_threshold_min:
                    continue   # gap too small

                # Find services that fit
                avail_services = svc_by_room_type.get(room["room_type"], pd.DataFrame())
                avail_services = avail_services[
                    avail_services["duration_min"] <= gap_min
                ]
                if avail_services.empty:
                    continue

                # Stochastic fill — model imperfect last-minute demand
                if random.random() > cfg.fill_rate:
                    gap_records.append({
                        "date": date, "room_id": room["room_id"],
                        "gap_start_min": gap_start, "gap_end_min": gap_end,
                        "gap_min": gap_min, "filled": False,
                        "revenue_gained": 0,
                    })
                    continue

                # Pick the highest-revenue service that fits
                svc = avail_services.sort_values("price", ascending=False).iloc[0]

                # Find a free provider
                candidates = PROVIDERS[
                    PROVIDERS["provider_id"].isin(svc["provider_ids"])
                ].sample(frac=1, random_state=fill_id)

                appt_start_dt = base_dt + timedelta(minutes=gap_start)
                appt_end_dt   = appt_start_dt + timedelta(minutes=int(svc["duration_min"]))

                chosen = None
                for _, prov in candidates.iterrows():
                    pid   = prov["provider_id"]
                    busy  = provider_busy.get(pid, [])
                    clash = any(
                        appt_start_dt < b_end and appt_end_dt > b_start
                        for b_start, b_end in busy
                    )
                    if not clash:
                        chosen = prov
                        break

                if chosen is None:
                    gap_records.append({
                        "date": date, "room_id": room["room_id"],
                        "gap_start_min": gap_start, "gap_end_min": gap_end,
                        "gap_min": gap_min, "filled": False,
                        "revenue_gained": 0,
                    })
                    continue

                # Register new provider slot
                pid = chosen["provider_id"]
                provider_busy.setdefault(pid, []).append((appt_start_dt, appt_end_dt))

                # Revenue = listed price (gap fills are priced at list rate)
                revenue = float(svc["price"])

                filled_records.append({
                    "fill_id":        f"FILL{fill_id:04d}",
                    "date":           date,
                    "room_id":        room["room_id"],
                    "room_name":      room["room_name"],
                    "provider_id":    chosen["provider_id"],
                    "provider_name":  chosen["name"],
                    "service_id":     svc["service_id"],
                    "service_name":   svc["service_name"],
                    "start_time":     appt_start_dt,
                    "end_time":       appt_end_dt,
                    "duration_min":   int(svc["duration_min"]),
                    "revenue":        revenue,
                    "gap_available_min": gap_min,
                })
                gap_records.append({
                    "date": date, "room_id": room["room_id"],
                    "gap_start_min": gap_start, "gap_end_min": gap_end,
                    "gap_min": gap_min, "filled": True,
                    "revenue_gained": revenue,
                })
                fill_id += 1

    filled_df  = pd.DataFrame(filled_records)
    gap_df     = pd.DataFrame(gap_records)

    # Daily uplift summary
    if not filled_df.empty:
        daily_uplift = (
            filled_df.groupby("date")
            .agg(
                fills_injected  = ("fill_id",  "count"),
                revenue_gained  = ("revenue",  "sum"),
                minutes_filled  = ("duration_min", "sum"),
            )
            .reset_index()
        )
    else:
        daily_uplift = pd.DataFrame(columns=["date","fills_injected","revenue_gained","minutes_filled"])

    total_revenue_gain = filled_df["revenue"].sum() if not filled_df.empty else 0
    total_gaps_found   = len(gap_df)
    total_gaps_filled  = gap_df["filled"].sum() if not gap_df.empty else 0

    return {
        "filled_slots":       filled_df,
        "gap_summary":        gap_df,
        "daily_uplift":       daily_uplift,
        "total_revenue_gain": round(total_revenue_gain, 2),
        "total_gaps_found":   int(total_gaps_found),
        "total_gaps_filled":  int(total_gaps_filled),
        "fill_rate_actual":   round(total_gaps_filled / max(total_gaps_found, 1), 4),
        "cfg":                cfg,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RISK-ADJUSTED OVERBOOKING MODEL
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class OverbookConfig:
    """Parameters for the overbooking model."""
    observed_noshow_rate:  float = 0.137   # from Phase 2 (13.7%)
    observed_cancel_rate:  float = 0.071   # from Phase 2 (7.1%)
    overbook_factors:      list  = field(default_factory=lambda: [0.05, 0.10, 0.15, 0.20])
    # Cost of an angry walk-in who can't be served (reputation / goodwill cost)
    walk_in_cost:          float = 150.0
    # Revenue collected when an overbooked slot is actually filled (no-show didn't arrive)
    avg_appt_revenue:      float = 303.0   # set at runtime


def simulate_overbooking(
    appointments: pd.DataFrame,
    cfg:          OverbookConfig,
) -> dict:
    """
    Monte-Carlo overbooking model.

    For each overbook_factor f, we simulate 10,000 days:
      - Base capacity C appointments booked
      - Overbook by factor f → schedule C × (1 + f) appointments
      - Each appointment independently shows up with p = 1 − no_show_rate − cancel_rate
      - If actual shows > C → overflow → each overflow triggers walk_in_cost penalty
      - If actual shows ≤ C → empty slots → each empty slot = lost revenue avoided
        (we already had the booking so no extra gain; the gain is the overbooked
         revenue that actually materialises)

    Revenue model per simulated day:
      revenue = shows_that_fit × avg_appt_revenue − overflow × walk_in_cost

    Returns
    -------
    dict with keys:
      scenario_summary   DataFrame — one row per overbook factor
      simulation_detail  DataFrame — raw Monte-Carlo results
      optimal_factor     scalar — factor with highest expected net revenue
    """
    N_SIM      = 10_000
    rng        = np.random.default_rng(42)
    show_prob  = 1.0 - cfg.observed_noshow_rate - cfg.observed_cancel_rate

    # Derive a reasonable "base daily capacity" from the data
    daily_appts = appointments.groupby("date")["appointment_id"].count()
    base_cap    = int(daily_appts.mean())   # avg scheduled per day

    records = []

    for factor in cfg.overbook_factors:
        overbooked_n = int(np.ceil(base_cap * (1 + factor)))
        extra_slots  = overbooked_n - base_cap

        # Simulate shows across N_SIM days
        shows = rng.binomial(n=overbooked_n, p=show_prob, size=N_SIM)

        # Revenue per simulation
        revenue_base  = np.minimum(shows, base_cap) * cfg.avg_appt_revenue
        overflow      = np.maximum(shows - base_cap, 0)
        penalty       = overflow * cfg.walk_in_cost
        net_revenue   = revenue_base - penalty

        # Baseline (no overbooking) for comparison
        baseline_shows   = rng.binomial(n=base_cap, p=show_prob, size=N_SIM)
        baseline_revenue = baseline_shows * cfg.avg_appt_revenue

        uplift = net_revenue - baseline_revenue

        records.append({
            "overbook_factor":      factor,
            "overbook_factor_pct":  f"+{factor*100:.0f}%",
            "base_capacity":        base_cap,
            "overbooked_slots":     overbooked_n,
            "extra_slots":          extra_slots,
            "mean_shows":           round(shows.mean(), 2),
            "overflow_prob":        round((overflow > 0).mean(), 4),
            "mean_overflow":        round(overflow.mean(), 4),
            "mean_net_revenue":     round(net_revenue.mean(), 2),
            "mean_baseline_rev":    round(baseline_revenue.mean(), 2),
            "mean_uplift":          round(uplift.mean(), 2),
            "uplift_5pct":          round(np.percentile(uplift,  5), 2),
            "uplift_95pct":         round(np.percentile(uplift, 95), 2),
            "mean_penalty":         round(penalty.mean(), 2),
            "revenue_std":          round(net_revenue.std(), 2),
            "sharpe_ratio":         round(uplift.mean() / (uplift.std() + 1e-9), 4),
        })

        # Store raw for density chart (sample 500 for size)
        for v in uplift[:500]:
            pass   # lightweight; aggregated above

    scenario_df = pd.DataFrame(records)
    optimal_row = scenario_df.loc[scenario_df["mean_uplift"].idxmax()]

    return {
        "scenario_summary": scenario_df,
        "optimal_factor":   optimal_row["overbook_factor"],
        "optimal_pct":      optimal_row["overbook_factor_pct"],
        "optimal_uplift":   optimal_row["mean_uplift"],
        "base_capacity":    base_cap,
        "cfg":              cfg,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — REVENUE SENSITIVITY ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def simulate_sensitivity(
    appointments:   pd.DataFrame,
    idle_time:      pd.DataFrame,
    leakage:        dict,
    uplift_targets: list[float] = None,
) -> dict:
    """
    Model revenue impact of incremental utilization improvements across levers:

    Levers modelled:
      A. Filling idle time  (gap-fill)
      B. Reducing no-show rate
      C. Reducing late-cancel rate
      D. Pricing optimisation (close listed→actual gap)

    For each lever × utilization uplift target (+5%, +10%) compute:
      - Additional revenue recovered
      - New total revenue
      - New utilization %
      - ROI estimate (revenue gain vs assumed intervention cost)

    Returns scenario_matrix DataFrame and lever_detail DataFrame.
    """
    if uplift_targets is None:
        uplift_targets = [0.05, 0.10]

    # ── Baseline scalars ────────────────────────────────────────────────
    base_revenue       = appointments["actual_revenue"].sum()
    operating_days     = appointments["date"].nunique()
    n_rooms            = len(ROOMS)
    total_avail_min    = operating_days * n_rooms * AVAILABLE_MIN_PER_DAY
    total_booked_min   = appointments["scheduled_duration_min"].sum()
    base_utilization   = total_booked_min / total_avail_min

    avg_rev_per_min    = leakage["avg_rev_per_min"]
    total_idle_min     = leakage["total_idle_min"]
    no_show_leakage    = leakage["no_show_leakage"]
    late_cancel_leakage= leakage["late_cancel_leakage"]
    discount_leakage   = leakage["discount_leakage"]

    # ── Lever definitions ────────────────────────────────────────────────
    levers = [
        {
            "lever":        "Gap-Fill (Idle Time)",
            "lever_id":     "A",
            "max_pool":     leakage["idle_leakage"],   # total idle leakage $
            "description":  "Fill idle room slots with last-minute or walk-in bookings",
        },
        {
            "lever":        "No-Show Reduction",
            "lever_id":     "B",
            "max_pool":     no_show_leakage,
            "description":  "Deposits + reminders reduce no-show rate",
        },
        {
            "lever":        "Late-Cancel Reduction",
            "lever_id":     "C",
            "max_pool":     late_cancel_leakage,
            "description":  "Stricter cancellation policy + waitlist automation",
        },
        {
            "lever":        "Pricing Optimisation",
            "lever_id":     "D",
            "max_pool":     discount_leakage,
            "description":  "Tighten discount controls; enforce listed pricing",
        },
    ]

    # Assumed one-time / monthly intervention costs (realistic estimates)
    intervention_costs = {"A": 800, "B": 300, "C": 200, "D": 150}

    rows = []
    for uplift in uplift_targets:
        for lv in levers:
            revenue_gain   = round(lv["max_pool"] * uplift / base_utilization * (base_utilization + uplift), 2)
            # Cap at available pool
            revenue_gain   = min(revenue_gain, lv["max_pool"])
            new_revenue    = round(base_revenue + revenue_gain, 2)
            new_util       = min(base_utilization + uplift, 1.0)
            monthly_cost   = intervention_costs[lv["lever_id"]]
            roi            = round((revenue_gain - monthly_cost) / monthly_cost * 100, 1)

            rows.append({
                "lever_id":       lv["lever_id"],
                "lever":          lv["lever"],
                "description":    lv["description"],
                "uplift_target":  uplift,
                "uplift_pct":     f"+{uplift*100:.0f}%",
                "max_pool_$":     round(lv["max_pool"], 2),
                "revenue_gain_$": revenue_gain,
                "new_revenue_$":  new_revenue,
                "base_util":      round(base_utilization, 4),
                "new_util":       round(new_util, 4),
                "monthly_cost_$": monthly_cost,
                "roi_pct":        roi,
            })

    scenario_matrix = pd.DataFrame(rows)

    # ── Combined scenario: all levers simultaneously ─────────────────────
    combo_rows = []
    for uplift in uplift_targets:
        total_gain  = scenario_matrix[scenario_matrix["uplift_target"] == uplift]["revenue_gain_$"].sum()
        total_cost  = sum(intervention_costs.values())
        combo_rows.append({
            "scenario":       f"All Levers  {'+'+str(int(uplift*100))+'%'}",
            "uplift_target":  uplift,
            "revenue_gain_$": round(total_gain, 2),
            "new_revenue_$":  round(base_revenue + total_gain, 2),
            "monthly_cost_$": total_cost,
            "roi_pct":        round((total_gain - total_cost) / total_cost * 100, 1),
            "new_util":       round(min(base_utilization + uplift, 1.0), 4),
        })
    combo_df = pd.DataFrame(combo_rows)

    return {
        "scenario_matrix": scenario_matrix,
        "combo_scenarios": combo_df,
        "base_revenue":    base_revenue,
        "base_utilization":base_utilization,
        "uplift_targets":  uplift_targets,
        "intervention_costs": intervention_costs,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════════════

def plot_gap_fill_daily_uplift(
    daily_kpis:   pd.DataFrame,
    daily_uplift: pd.DataFrame,
) -> str:
    """
    Stacked area: baseline revenue + gap-fill uplift vs theoretical maximum.
    """
    base = daily_kpis.sort_values("date")[["date","actual_revenue","theoretical_max_revenue"]].copy()
    base["date"] = pd.to_datetime(base["date"])

    uplift = daily_uplift.copy()
    if not uplift.empty:
        uplift["date"] = pd.to_datetime(uplift["date"])
        base = base.merge(uplift[["date","revenue_gained"]], on="date", how="left")
    else:
        base["revenue_gained"] = 0

    base["revenue_gained"]   = base["revenue_gained"].fillna(0)
    base["optimised_revenue"] = base["actual_revenue"] + base["revenue_gained"]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.fill_between(base["date"], 0, base["actual_revenue"],
                    alpha=0.70, color=PALETTE["primary"], label="Baseline Revenue")
    ax.fill_between(base["date"], base["actual_revenue"], base["optimised_revenue"],
                    alpha=0.75, color=PALETTE["success"], label="Gap-Fill Gain")
    ax.fill_between(base["date"], base["optimised_revenue"], base["theoretical_max_revenue"],
                    alpha=0.20, color=PALETTE["danger"],  label="Remaining Leakage")

    ax.plot(base["date"], base["theoretical_max_revenue"],
            color=PALETTE["neutral"], linewidth=1.6, linestyle="--", label="Theoretical Max")

    ax.set_title("Gap-Fill Simulation: Daily Revenue Uplift", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9, loc="lower right")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    sns.despine()
    fig.tight_layout()
    return _save(fig, "chart_08_gap_fill_daily_uplift.png")


def plot_gap_fill_room_breakdown(gap_summary: pd.DataFrame) -> str:
    """
    Grouped bar chart: gaps found vs gaps filled per room.
    """
    from phase1_data_engine import ROOMS as R
    room_labels = R.set_index("room_id")["room_name"].to_dict()

    agg = (
        gap_summary
        .groupby("room_id")
        .agg(gaps_found=("gap_min","count"), gaps_filled=("filled","sum"))
        .reset_index()
    )
    agg["room_name"]    = agg["room_id"].map(room_labels)
    agg["gaps_unfilled"]= agg["gaps_found"] - agg["gaps_filled"]

    x   = np.arange(len(agg))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, agg["gaps_found"],   w, label="Gaps Found",   color=PALETTE["neutral"],  edgecolor="white")
    ax.bar(x + w/2, agg["gaps_filled"],  w, label="Gaps Filled",  color=PALETTE["success"],  edgecolor="white")

    for i, (_, row) in enumerate(agg.iterrows()):
        pct = row["gaps_filled"] / max(row["gaps_found"], 1) * 100
        ax.text(i + w/2, row["gaps_filled"] + 0.3, f"{pct:.0f}%",
                ha="center", fontsize=8.5, fontweight="bold", color=PALETTE["success"])

    ax.set_xticks(x)
    ax.set_xticklabels(agg["room_name"], rotation=15, ha="right")
    ax.set_title("Gap-Fill Simulation: Gaps Found vs Filled per Room", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Gaps")
    ax.legend(fontsize=9)
    sns.despine()
    fig.tight_layout()
    return _save(fig, "chart_09_gap_fill_room_breakdown.png")


def plot_overbooking_scenarios(ob_results: dict) -> str:
    """
    Two-panel chart:
      Left  — Mean daily revenue uplift with 5th/95th CI ribbon per overbook factor
      Right — Overflow probability vs overbook factor (risk curve)
    """
    df = ob_results["scenario_summary"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: uplift bar + CI ─────────────────────────────────────────────
    colors = [PALETTE["success"] if v > 0 else PALETTE["danger"] for v in df["mean_uplift"]]
    bars   = ax1.bar(df["overbook_factor_pct"], df["mean_uplift"], color=colors,
                     edgecolor="white", width=0.45, zorder=3)

    # CI error bars
    yerr_lo = df["mean_uplift"] - df["uplift_5pct"]
    yerr_hi = df["uplift_95pct"] - df["mean_uplift"]
    ax1.errorbar(
        df["overbook_factor_pct"], df["mean_uplift"],
        yerr=[yerr_lo, yerr_hi],
        fmt="none", color="#333", capsize=5, linewidth=1.5, zorder=4,
    )

    for bar, (_, row) in zip(bars, df.iterrows()):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(yerr_hi) * 0.05,
                 f"${row['mean_uplift']:+,.0f}",
                 ha="center", fontsize=8.5, fontweight="bold")

    ax1.axhline(0, color="#999", linewidth=1)
    ax1.set_title("Mean Daily Revenue Uplift\n(vs No-Overbook Baseline)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Overbook Factor")
    ax1.set_ylabel("Revenue Uplift ($)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    sns.despine(ax=ax1)

    # ── Right: overflow probability risk curve ────────────────────────────
    ax2.plot(df["overbook_factor_pct"], df["overflow_prob"] * 100,
             color=PALETTE["danger"], marker="o", linewidth=2.2, markersize=7)
    ax2.fill_between(range(len(df)), df["overflow_prob"] * 100,
                     alpha=0.12, color=PALETTE["danger"])

    for i, (_, row) in enumerate(df.iterrows()):
        ax2.text(i, row["overflow_prob"] * 100 + 0.8,
                 f"{row['overflow_prob']*100:.1f}%",
                 ha="center", fontsize=8.5, color=PALETTE["danger"], fontweight="bold")

    ax2.set_title("Overflow Probability by Overbook Factor\n(Risk Curve)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Overbook Factor")
    ax2.set_ylabel("Overflow Probability (%)")
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df["overbook_factor_pct"])
    sns.despine(ax=ax2)

    # Mark optimal factor
    opt_idx = df["mean_uplift"].idxmax()
    ax1.get_children()[opt_idx].set_edgecolor("#1a1a1a")
    ax1.get_children()[opt_idx].set_linewidth(2)

    fig.suptitle("Risk-Adjusted Overbooking Simulation (10,000 Monte-Carlo Days)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "chart_10_overbooking_scenarios.png")


def plot_sensitivity_matrix(sensitivity: dict) -> str:
    """
    Heatmap of revenue gain ($) by lever × uplift scenario.
    """
    df = sensitivity["scenario_matrix"]

    pivot = df.pivot(index="lever", columns="uplift_pct", values="revenue_gain_$")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=",.0f",
        cmap="YlGn", linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Revenue Gain ($)"},
    )
    ax.set_title("Sensitivity Matrix: Revenue Gain ($) by Lever × Uplift Target",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Utilization Uplift Target")
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return _save(fig, "chart_11_sensitivity_matrix.png")


def plot_sensitivity_roi(sensitivity: dict) -> str:
    """
    Grouped bar chart: ROI % by lever for +5% and +10% scenarios.
    """
    df = sensitivity["scenario_matrix"]
    p5  = df[df["uplift_target"] == 0.05].sort_values("lever_id")
    p10 = df[df["uplift_target"] == 0.10].sort_values("lever_id")

    x   = np.arange(len(p5))
    w   = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))

    bars5  = ax.bar(x - w/2, p5["roi_pct"],  w, label="+5% Uplift",  color=PALETTE["primary"], edgecolor="white")
    bars10 = ax.bar(x + w/2, p10["roi_pct"], w, label="+10% Uplift", color=PALETTE["accent"],  edgecolor="white")

    for bars in [bars5, bars10]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 5,
                    f"{h:,.0f}%", ha="center", fontsize=8, fontweight="bold")

    ax.axhline(0, color="#aaa", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(p5["lever"], rotation=10, ha="right")
    ax.set_title("ROI by Optimization Lever (Revenue Gain / Monthly Intervention Cost)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("ROI (%)")
    ax.legend(fontsize=9)
    sns.despine()
    fig.tight_layout()
    return _save(fig, "chart_12_sensitivity_roi.png")


def plot_combined_waterfall(
    base_revenue: float,
    gap_gain:     float,
    ob_gain:      float,
    sensitivity:  dict,
) -> str:
    """
    Waterfall chart: Baseline → Gap-Fill → Overbooking → Sensitivity scenarios.
    """
    s5  = sensitivity["combo_scenarios"][sensitivity["combo_scenarios"]["uplift_target"] == 0.05]
    s10 = sensitivity["combo_scenarios"][sensitivity["combo_scenarios"]["uplift_target"] == 0.10]

    s5_gain  = float(s5["revenue_gain_$"].iloc[0]) if not s5.empty else 0
    s10_gain = float(s10["revenue_gain_$"].iloc[0]) if not s10.empty else 0

    labels = ["Baseline\nRevenue", "Gap-Fill\nGain", "Overbook\nGain",
              "+5% Sensitivity\n(All Levers)", "+10% Sensitivity\n(All Levers)"]
    values = [base_revenue, gap_gain, ob_gain, s5_gain, s10_gain]

    running = [base_revenue]
    for v in values[1:]:
        running.append(running[-1] + v)

    bar_bottoms = [0] + running[:-1]

    colors = [PALETTE["primary"]] + [PALETTE["success"]] * (len(values) - 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (label, val, bottom, color) in enumerate(zip(labels, values, bar_bottoms, colors)):
        ax.bar(i, val, bottom=bottom, color=color, edgecolor="white", width=0.55, zorder=3)
        # Running total annotation
        ax.text(i, running[i] + base_revenue * 0.005, f"${running[i]:,.0f}",
                ha="center", fontsize=9, fontweight="bold")
        # Delta annotation
        if i > 0:
            ax.text(i, bottom + val / 2, f"+${val:,.0f}",
                    ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    # Connector lines
    for i in range(len(labels) - 1):
        ax.plot([i + 0.28, i + 0.72], [running[i], running[i]],
                color="#aaa", linewidth=1, linestyle="--", zorder=2)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Optimization Waterfall: Revenue Impact by Strategy",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_ylabel("Cumulative Revenue ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_ylim(0, running[-1] * 1.12)
    sns.despine()
    fig.tight_layout()
    return _save(fig, "chart_13_optimization_waterfall.png")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EXECUTIVE SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

def print_phase3_summary(
    gap_results:  dict,
    ob_results:   dict,
    sensitivity:  dict,
    base_revenue: float,
) -> None:
    SEP = "═" * 64
    sep = "─" * 64

    print(f"\n{SEP}")
    print("  MED SPA OPTIMIZATION SIMULATION — EXECUTIVE SUMMARY  (Phase 3)")
    print(SEP)

    # ── Gap Fill ─────────────────────────────────────────────────────────
    print(f"\n  1. GAP-FILL ALGORITHM")
    print(sep)
    print(f"  Gaps detected          : {gap_results['total_gaps_found']:,}")
    print(f"  Gaps filled            : {gap_results['total_gaps_filled']:,}"
          f"  ({gap_results['fill_rate_actual']:.1%} fill rate)")
    print(f"  Revenue gained         : ${gap_results['total_revenue_gain']:,.2f}")
    print(f"  Revenue uplift         : {gap_results['total_revenue_gain']/base_revenue:.1%} on baseline")

    # ── Overbooking ──────────────────────────────────────────────────────
    ob_df = ob_results["scenario_summary"]
    print(f"\n  2. RISK-ADJUSTED OVERBOOKING  (10,000 Monte-Carlo simulations/scenario)")
    print(sep)
    print(f"  {'Factor':<10} {'Extra Slots':>11} {'Mean Uplift':>13} {'Overflow Prob':>14} {'Sharpe':>8}")
    print(f"  {'-'*8:<10} {'-'*9:>11} {'-'*11:>13} {'-'*12:>14} {'-'*6:>8}")
    for _, row in ob_df.iterrows():
        marker = " ◀ OPTIMAL" if row["overbook_factor"] == ob_results["optimal_factor"] else ""
        print(f"  {row['overbook_factor_pct']:<10} {row['extra_slots']:>11}"
              f" ${row['mean_uplift']:>11,.0f} {row['overflow_prob']*100:>13.1f}%"
              f" {row['sharpe_ratio']:>8.3f}{marker}")

    # ── Sensitivity ──────────────────────────────────────────────────────
    sm = sensitivity["scenario_matrix"]
    print(f"\n  3. REVENUE SENSITIVITY ANALYSIS")
    print(sep)
    print(f"  {'Lever':<28} {'Max Pool':>10} {'Gain +5%':>11} {'Gain +10%':>11} {'ROI +5%':>9} {'ROI +10%':>10}")
    print(f"  {'-'*26:<28} {'-'*8:>10} {'-'*9:>11} {'-'*9:>11} {'-'*7:>9} {'-'*8:>10}")
    for lever_id in ["A", "B", "C", "D"]:
        r5  = sm[(sm["lever_id"] == lever_id) & (sm["uplift_target"] == 0.05)].iloc[0]
        r10 = sm[(sm["lever_id"] == lever_id) & (sm["uplift_target"] == 0.10)].iloc[0]
        print(f"  {r5['lever']:<28} ${r5['max_pool_$']:>9,.0f}"
              f" ${r5['revenue_gain_$']:>9,.0f} ${r10['revenue_gain_$']:>9,.0f}"
              f" {r5['roi_pct']:>8.0f}% {r10['roi_pct']:>9.0f}%")

    # Combo
    combo = sensitivity["combo_scenarios"]
    print(f"\n  COMBINED SCENARIO (All Levers)")
    for _, row in combo.iterrows():
        print(f"  {row['scenario']:<32}  Gain: ${row['revenue_gain_$']:>9,.0f}"
              f"  New Revenue: ${row['new_revenue_$']:>9,.0f}"
              f"  ROI: {row['roi_pct']:,.0f}%")

    # ── Recommendations ──────────────────────────────────────────────────
    best_lever = sm.loc[sm[sm["uplift_target"]==0.10]["roi_pct"].idxmax()]
    print(f"\n  KEY RECOMMENDATIONS")
    print(sep)
    print(f"  ① IMPLEMENT GAP-FILL BOOKING ENGINE")
    print(f"     → ${gap_results['total_revenue_gain']:,.0f} recoverable over 30 days "
          f"at {gap_results['fill_rate_actual']:.0%} fill rate.")
    print(f"     → Deploy automated SMS 'open slot' alerts 2–4 hrs before idle windows.")
    print()
    print(f"  ② OVERBOOK AT {ob_results['optimal_pct']} (OPTIMAL FACTOR)")
    print(f"     → Expected daily uplift: +${ob_results['optimal_uplift']:,.0f}")
    ob_row = ob_df[ob_df["overbook_factor"] == ob_results["optimal_factor"]].iloc[0]
    print(f"     → Overflow risk: {ob_row['overflow_prob']*100:.1f}% of days — "
          f"mitigate with waitlist to absorb overflow gracefully.")
    print()
    print(f"  ③ PRIORITISE LEVER: {best_lever['lever'].upper()}")
    print(f"     → Highest ROI at +10% uplift: {best_lever['roi_pct']:,.0f}%")
    print(f"     → Recover up to ${best_lever['revenue_gain_$']:,.0f} with ${best_lever['monthly_cost_$']} monthly spend.")
    print(f"\n{SEP}\n")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def run_phase3(
    appointments: "pd.DataFrame",
    idle_time:    "pd.DataFrame",
    daily_kpis:   "pd.DataFrame",
    leakage:      dict,
) -> dict:
    """
    Orchestrates Phase 3 simulations.

    Parameters
    ----------
    appointments : DataFrame produced by Phase 1 generate_all_data()
    idle_time    : DataFrame produced by Phase 1 generate_all_data()
    daily_kpis   : DataFrame produced by Phase 2 compute_daily_kpis()
    leakage      : dict produced by Phase 2 compute_leakage()

    All input data must be supplied by the caller — Phase 3 does NOT
    regenerate Phase 1 data or recompute Phase 2 analytics.

    Returns dict with all simulation results for Phase 4 dashboard.
    """
    print("\n🔬  Running Phase 3 — Optimization Simulation...")

    # ── 1. Gap-Fill ──────────────────────────────────────────────────────
    print("  ⚙️   Gap-Fill Algorithm...")
    gap_cfg = GapFillConfig(avg_revenue_per_min=leakage["avg_rev_per_min"])
    gap_results = simulate_gap_filling(appointments, idle_time, gap_cfg)

    # ── 2. Overbooking ───────────────────────────────────────────────────
    print("  ⚙️   Overbooking Monte-Carlo...")
    completed       = appointments[appointments["status"] == "Completed"]
    avg_appt_rev    = completed["actual_revenue"].mean()
    ob_cfg = OverbookConfig(avg_appt_revenue=round(avg_appt_rev, 2))
    ob_results = simulate_overbooking(appointments, ob_cfg)

    # ── 3. Sensitivity ───────────────────────────────────────────────────
    print("  ⚙️   Sensitivity Analysis...")
    sensitivity = simulate_sensitivity(appointments, idle_time, leakage)

    # ── Charts ───────────────────────────────────────────────────────────
    print("  🎨  Generating charts...")
    chart_paths = {
        "gap_fill_daily":     plot_gap_fill_daily_uplift(daily_kpis, gap_results["daily_uplift"]),
        "gap_fill_rooms":     plot_gap_fill_room_breakdown(gap_results["gap_summary"]),
        "overbooking":        plot_overbooking_scenarios(ob_results),
        "sensitivity_matrix": plot_sensitivity_matrix(sensitivity),
        "sensitivity_roi":    plot_sensitivity_roi(sensitivity),
        "waterfall":          plot_combined_waterfall(
                                  sensitivity["base_revenue"],
                                  gap_results["total_revenue_gain"],
                                  ob_results["optimal_uplift"],
                                  sensitivity,
                              ),
    }
    for name, path in chart_paths.items():
        print(f"     ✅  {name:<22}  →  {os.path.basename(path)}")

    # ── Executive Summary ────────────────────────────────────────────────
    print_phase3_summary(gap_results, ob_results, sensitivity, sensitivity["base_revenue"])

    return {
        "gap_results":   gap_results,
        "ob_results":    ob_results,
        "sensitivity":   sensitivity,
        "charts":        chart_paths,
        # Pass-through for Phase 4
        "appointments":  appointments,
        "idle_time":     idle_time,
        "daily_kpis":    daily_kpis,
        "leakage":       leakage,
    }


if __name__ == "__main__":
    # ── Produce Phase 1 data ─────────────────────────────────────────────
    raw          = generate_all_data()
    appointments = raw["appointments"]
    idle_time    = raw["idle_time"]

    # ── Produce Phase 2 analytics ────────────────────────────────────────
    from phase2_data_analytics import compute_daily_kpis, compute_leakage
    leakage    = compute_leakage(appointments, idle_time)
    daily_kpis = compute_daily_kpis(appointments, idle_time)

    # ── Run Phase 3 with pre-computed inputs ─────────────────────────────
    results = run_phase3(appointments, idle_time, daily_kpis, leakage)
