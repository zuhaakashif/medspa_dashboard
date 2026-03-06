# 💆 MedSpa Revenue Leakage & Utilization Intelligence Dashboard

A full-stack healthcare operations analytics platform that simulates, quantifies, and optimizes revenue leakage in med spa scheduling. Built as a proof-of-concept for AI-driven scheduling intelligence in the aesthetic medicine industry.

**[🚀 Live Demo](https://medspadashboard-test.streamlit.app/)** &nbsp;|&nbsp; Built with Python, Pandas, Matplotlib, Seaborn, Streamlit

---

## 📌 Overview

Most med spas operate at 60–70% capacity without knowing it. This platform identifies exactly where revenue is leaking, quantifies the dollar impact, and simulates the ROI of specific interventions — the same analytical problem that AI scheduling optimization products are built to solve.

The system is built in four phases:

| Phase | Description |
|-------|-------------|
| **Phase 1** | Synthetic operations engine — generates 30 days of realistic scheduling data |
| **Phase 2** | Analytics engine — KPIs, utilization metrics, leakage quantification |
| **Phase 3** | Optimization simulation — gap-fill algorithm, overbooking model, sensitivity analysis |
| **Phase 4** | Streamlit dashboard — interactive executive decision-support interface |

---

## 📊 Dashboard Sections

### 1. Executive Operations Overview
Six top-level KPI cards pulling live from the data pipeline:
- Total Appointments, Utilization Rate, Total Revenue
- Revenue Leakage, No-Show Rate, Avg Revenue per Visit

Room utilization heatmap, daily revenue vs theoretical maximum, provider performance, service mix, and peak hour analysis.

### 2. Revenue Leakage Analysis
Quantifies four leakage channels with a donut chart and sortable source table:
- **Idle Time** — unbilled room capacity (typically ~80% of total leakage)
- **No-Shows** — full listed price lost
- **Late Cancellations** — listed price minus fees collected
- **Pricing Discounts** — gap between listed and actual price on completed appointments

### 3. Scheduling Inefficiency Analysis
- Room utilization heatmap across the full 30-day window
- Revenue by hour of day and day of week
- Auto-computed insights: peak hour, trough hour, worst-performing room, utilization volatility

### 4. Optimization Simulation
Three simulation engines with four interactive tabs:

**Gap-Fill Algorithm** — scans every (date × room) timeline for idle windows ≥ 30 min, respects provider availability, and injects synthetic fill appointments at a configurable fill rate.

**Risk-Adjusted Overbooking** — 10,000 Monte Carlo simulations per overbook factor (+5% to +20%), modelling show probability against observed no-show rates. Returns mean uplift, overflow probability, and Sharpe ratio per scenario.

**Waterfall Chart** — cumulative revenue impact across all strategies combined.

**Interactive Simulator** — four sliders (no-show rate, overbooking %, demand growth, gap-fill success rate) that live-compute projected revenue, recovered revenue, and projected utilization.

### 5. Sensitivity Modeling
Revenue gain heatmap by lever × uplift target, ROI bar chart, and a multiselect scenario explorer that recalculates combined ROI live.

### 6. AI Operational Recommendations
Programmatically generated recommendations with severity badges (Critical / High / Medium / Low), specific dollar figures derived from the simulation, a prioritized action table, and combined ROI estimate.

---

## 🔑 Key Findings (Synthetic Data)

| Metric | Value |
|--------|-------|
| Total Revenue (30 days) | $182,906 |
| Total Revenue Leakage | $238,314 |
| Idle Time Leakage | $188,998 (79.3%) |
| No-Show Leakage | $31,520 (13.2%) |
| Gap-Fill Recovery Potential | $7,750 at 70% fill rate |
| No-Show Reduction ROI | 470% at +5% uplift |
| All-Lever Combined ROI | 1,826% at +10% uplift |

---

## 🗂️ Project Structure

```
medspa_dashboard/
├── app.py                      # Streamlit dashboard (Phase 4)
├── phase1_data_engine.py       # Synthetic operations data generator
├── phase2_data_analytics.py    # KPI computation & leakage analysis
├── phase3_optimization.py      # Optimization simulation engines
├── requirements.txt            # Python dependencies
└── .streamlit/
    └── config.toml             # Theme & server configuration
```

---

## ⚙️ Running Locally

**Install dependencies:**
```bash
pip install streamlit pandas numpy matplotlib seaborn
```

**Run the dashboard:**
```bash
streamlit run app.py
```

**Run individual phases standalone:**
```bash
python phase1_data_engine.py    # Generate data, print summary
python phase2_data_analytics.py # Run analytics, save charts
python phase3_optimization.py   # Run simulations, save charts
```

**Using Phase 3 as a module:**
```python
from phase1_data_engine import generate_all_data
from phase2_data_analytics import compute_leakage, compute_daily_kpis
from phase3_optimization import run_phase3

raw        = generate_all_data()
leakage    = compute_leakage(raw["appointments"], raw["idle_time"])
daily_kpis = compute_daily_kpis(raw["appointments"], raw["idle_time"])

results = run_phase3(raw["appointments"], raw["idle_time"], daily_kpis, leakage)
```

---

## 🏗️ Architecture Notes

**Phase 3 data contract** — `run_phase3()` accepts pre-computed DataFrames from Phases 1 and 2. It does not regenerate data internally, making it composable and testable in isolation.

**Scheduling integrity** — the synthetic engine enforces zero room overlaps, zero provider double-bookings, and full business-hours compliance by construction using a sequential cursor-based scheduler with a shared daily provider registry.

**Reproducibility** — all randomness is seeded (`RANDOM_SEED = 42`), so every run produces identical data.

**Real data integration** — the architecture is designed to replace `generate_all_data()` with a connector to any PMS/EHR system (Mindbody, Jane App, Vagaro, etc.) without changing any downstream analytics or dashboard code.

---

## 🧰 Tech Stack

| Layer | Tools |
|-------|-------|
| Data generation | Python, Pandas, NumPy |
| Analytics | Pandas, NumPy |
| Simulation | NumPy (Monte Carlo), custom gap-fill algorithm |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## 📬 Contact

Built by **Zuha Fatima**  
[GitHub](https://github.com/zuhaakashif)

---

*Built as a proof-of-concept for AI-driven scheduling intelligence in the aesthetic medicine industry. All data is synthetically generated — no real patient or business data is used.*
