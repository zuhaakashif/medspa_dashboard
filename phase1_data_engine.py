"""
Phase 1 — Synthetic Operations Engine
Med Spa Revenue Leakage & Utilization Intelligence Sandbox
Generates 30-day realistic operational data as pandas DataFrames
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ── Seed for reproducibility ──────────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# MASTER CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
START_DATE      = datetime(2026, 2, 1)
SIM_DAYS        = 30
BUSINESS_OPEN   = 9    # 9 AM
BUSINESS_CLOSE  = 19   # 7 PM
SLOT_MINUTES    = 30   # scheduling granularity
NO_SHOW_RATE    = 0.12 # 12% industry-typical no-show rate
LATE_CANCEL_RATE= 0.07 # 7% late cancellations (< 24h)

# ─────────────────────────────────────────────────────────────────────────────
# STATIC ENTITIES
# ─────────────────────────────────────────────────────────────────────────────

ROOMS = pd.DataFrame([
    {"room_id": "R01", "room_name": "Treatment Room 1", "room_type": "Standard",    "hourly_cost": 25},
    {"room_id": "R02", "room_name": "Treatment Room 2", "room_type": "Standard",    "hourly_cost": 25},
    {"room_id": "R03", "room_name": "Laser Suite",      "room_type": "Specialized", "hourly_cost": 45},
    {"room_id": "R04", "room_name": "VIP Lounge",       "room_type": "Premium",     "hourly_cost": 60},
])

PROVIDERS = pd.DataFrame([
    {"provider_id": "P01", "name": "Dr. Sarah Chen",    "role": "Medical Director",   "specialties": ["Botox", "Filler", "Laser"], "hourly_rate": 120},
    {"provider_id": "P02", "name": "Jenna Mills, NP",   "role": "Nurse Practitioner", "specialties": ["Botox", "Filler", "IV Therapy"], "hourly_rate": 85},
    {"provider_id": "P03", "name": "Ava Torres, RN",    "role": "Aesthetic Nurse",    "specialties": ["HydraFacial", "Chemical Peel", "Microneedling"], "hourly_rate": 70},
    {"provider_id": "P04", "name": "Marcus Webb, LE",   "role": "Lead Esthetician",   "specialties": ["HydraFacial", "Chemical Peel", "Waxing"], "hourly_rate": 55},
])

SERVICES = pd.DataFrame([
    {"service_id": "S01", "service_name": "Botox",            "duration_min": 30,  "price": 350,  "required_room_type": ["Standard","Premium"],              "provider_ids": ["P01","P02"]},
    {"service_id": "S02", "service_name": "Dermal Filler",    "duration_min": 60,  "price": 650,  "required_room_type": ["Standard","Premium"],              "provider_ids": ["P01","P02"]},
    {"service_id": "S03", "service_name": "HydraFacial",      "duration_min": 60,  "price": 250,  "required_room_type": ["Standard","Premium"],              "provider_ids": ["P03","P04"]},
    {"service_id": "S04", "service_name": "Chemical Peel",    "duration_min": 45,  "price": 200,  "required_room_type": ["Standard","Premium"],              "provider_ids": ["P03","P04"]},
    {"service_id": "S05", "service_name": "Laser Hair Removal","duration_min": 60, "price": 300,  "required_room_type": ["Specialized"],                     "provider_ids": ["P01","P03"]},
    {"service_id": "S06", "service_name": "Laser Skin Resurfacing","duration_min": 90,"price": 500,"required_room_type": ["Specialized"],                    "provider_ids": ["P01"]},
    {"service_id": "S07", "service_name": "Microneedling",    "duration_min": 60,  "price": 350,  "required_room_type": ["Standard","Premium"],              "provider_ids": ["P03"]},
    {"service_id": "S08", "service_name": "IV Therapy",       "duration_min": 45,  "price": 180,  "required_room_type": ["Standard","Premium","Specialized"],"provider_ids": ["P02"]},
    {"service_id": "S09", "service_name": "Waxing",           "duration_min": 30,  "price": 80,   "required_room_type": ["Standard"],                        "provider_ids": ["P04"]},
    {"service_id": "S10", "service_name": "VIP Package",      "duration_min": 120, "price": 950,  "required_room_type": ["Premium"],                         "provider_ids": ["P01","P02"]},
])

# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _assign_status(price: float) -> tuple[str, float, int]:
    """
    Randomly assign appointment status and compute actual revenue / duration.
    Returns (status, actual_revenue, actual_duration_min).
    """
    rand = random.random()
    if rand < NO_SHOW_RATE:
        return "No-Show", 0.0, 0
    elif rand < NO_SHOW_RATE + LATE_CANCEL_RATE:
        return "Late Cancel", round(price * 0.25, 2), 0   # 25% cancellation fee
    else:
        return "Completed", round(price * random.uniform(0.90, 1.05), 2), -1  # -1 = use svc duration


def _build_room_schedule(
    date: datetime,
    room: pd.Series,
    provider_busy: dict[str, list[tuple[datetime, datetime]]],
    appt_counter: list[int],
    service_weights: list[float],
) -> list[dict]:
    """
    Fill one room's day with appointments using sequential scheduling.

    Rules enforced:
    - Appointments are placed back-to-back with a turnover gap (5–10 min).
    - 25 % chance of an additional idle gap (10–45 min) before next appointment.
    - A service is only scheduled when a compatible, currently-free provider exists.
    - Every appointment ends at or before BUSINESS_CLOSE.
    - No room overlap by construction (cursor never goes backwards).
    - No provider double-booking (checked against provider_busy registry).
    """
    day_close = date.replace(hour=BUSINESS_CLOSE, minute=0, second=0, microsecond=0)
    cursor    = date.replace(hour=BUSINESS_OPEN,  minute=0, second=0, microsecond=0)
    records   = []

    # Filter services compatible with this room type
    compatible_services = SERVICES[
        SERVICES["required_room_type"].apply(lambda rt: room["room_type"] in rt)
    ]
    if compatible_services.empty:
        return records

    svc_weights_filtered = [
        service_weights[int(row["service_id"][1:]) - 1]   # S01→idx 0, S10→idx 9
        for _, row in compatible_services.iterrows()
    ]

    while cursor < day_close:
        # ── Optional idle gap (25 % probability) ──────────────────────────
        if random.random() < 0.25:
            cursor += timedelta(minutes=random.randint(10, 45))
            if cursor >= day_close:
                break

        # ── Pick a service that fits in remaining time ─────────────────────
        remaining_min = (day_close - cursor).total_seconds() / 60
        eligible = compatible_services[
            compatible_services["duration_min"] <= remaining_min
        ]
        if eligible.empty:
            break

        # Rebuild weights for eligible subset
        eligible_weights = [
            service_weights[int(row["service_id"][1:]) - 1]
            for _, row in eligible.iterrows()
        ]

        svc = eligible.sample(1, weights=eligible_weights).iloc[0]

        # ── Find a free provider ───────────────────────────────────────────
        appt_start = cursor
        appt_end   = appt_start + timedelta(minutes=int(svc["duration_min"]))

        candidate_providers = PROVIDERS[
            PROVIDERS["provider_id"].isin(svc["provider_ids"])
        ].sample(frac=1)   # shuffle to avoid always picking same provider

        chosen_provider = None
        for _, prov in candidate_providers.iterrows():
            pid = prov["provider_id"]
            busy_slots = provider_busy.get(pid, [])
            # Check no overlap: new [start, end) must not intersect any existing slot
            conflict = any(
                appt_start < busy_end and appt_end > busy_start
                for busy_start, busy_end in busy_slots
            )
            if not conflict:
                chosen_provider = prov
                break

        if chosen_provider is None:
            # No free provider — advance cursor by one turnover unit and retry
            cursor += timedelta(minutes=random.randint(5, 10))
            continue

        # ── Register provider as busy ──────────────────────────────────────
        pid = chosen_provider["provider_id"]
        provider_busy.setdefault(pid, []).append((appt_start, appt_end))

        # ── Assign status ──────────────────────────────────────────────────
        status, actual_revenue, actual_duration = _assign_status(svc["price"])
        if actual_duration == -1:          # sentinel for Completed
            actual_duration = int(svc["duration_min"])

        appt_counter[0] += 1
        records.append({
            "appointment_id":         f"APT{appt_counter[0]:04d}",
            "date":                   date.date(),
            "day_of_week":            date.strftime("%A"),
            "start_time":             appt_start,
            "end_time":               appt_end,
            "scheduled_duration_min": int(svc["duration_min"]),
            "actual_duration_min":    actual_duration,
            "room_id":                room["room_id"],
            "room_name":              room["room_name"],
            "room_type":              room["room_type"],
            "provider_id":            chosen_provider["provider_id"],
            "provider_name":          chosen_provider["name"],
            "service_id":             svc["service_id"],
            "service_name":           svc["service_name"],
            "listed_price":           svc["price"],
            "actual_revenue":         actual_revenue,
            "status":                 status,
        })

        # ── Advance cursor: end of appointment + turnover gap ──────────────
        turnover = random.randint(5, 10)
        cursor   = appt_end + timedelta(minutes=turnover)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# APPOINTMENT GENERATOR  (refactored — strict scheduling integrity)
# ─────────────────────────────────────────────────────────────────────────────
def generate_appointments() -> pd.DataFrame:
    """
    Simulate 30 days of appointment bookings with strict scheduling integrity:
    - Sequential room scheduling (no room overlap by construction).
    - Provider double-booking prevented via shared daily busy registry.
    - Day-of-week demand variation controls room activation probability.
    - No-show / late-cancel logic preserved.
    """
    # Day-of-week demand multipliers (0=Mon … 6=Sun; spa closed Sun)
    dow_demand    = {0: 0.70, 1: 0.85, 2: 1.00, 3: 1.10, 4: 1.20, 5: 0.90, 6: 0.00}
    service_weights = [0.20, 0.12, 0.18, 0.12, 0.08, 0.05, 0.07, 0.06, 0.07, 0.05]

    all_records:  list[dict] = []
    appt_counter: list[int]  = [0]   # mutable int passed by reference into helper

    for day_offset in range(SIM_DAYS):
        date = START_DATE + timedelta(days=day_offset)
        dow  = date.weekday()

        if dow_demand[dow] == 0.0:      # closed Sunday
            continue

        demand = dow_demand[dow]

        # Provider busy registry is reset each day (shared across all rooms)
        provider_busy: dict[str, list[tuple[datetime, datetime]]] = {}

        for _, room in ROOMS.iterrows():
            # Demand multiplier gates whether a room opens at all today.
            # Lower-demand days may leave some rooms idle.
            if random.random() > demand:
                continue                # room sits dark today

            day_records = _build_room_schedule(
                date, room, provider_busy, appt_counter, service_weights
            )
            all_records.extend(day_records)

    return pd.DataFrame(all_records) if all_records else pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# IDLE TIME CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────
def compute_idle_time(appointments: pd.DataFrame) -> pd.DataFrame:
    """
    For each (date, room) pair compute idle minutes = available minutes − booked minutes.
    Booked = sum of scheduled durations for Completed/Late-Cancel appointments.
    """
    total_available_min = (BUSINESS_CLOSE - BUSINESS_OPEN) * 60  # 600 min

    booked = (
        appointments[appointments["status"].isin(["Completed", "Late Cancel"])]
        .groupby(["date", "room_id"])["scheduled_duration_min"]
        .sum()
        .reset_index()
        .rename(columns={"scheduled_duration_min": "booked_min"})
    )

    # cross-join all dates × rooms to capture zero-booking days
    all_dates = appointments["date"].unique()
    index = pd.MultiIndex.from_product([all_dates, ROOMS["room_id"]], names=["date","room_id"])
    full = pd.DataFrame(index=index).reset_index()

    merged = full.merge(booked, on=["date","room_id"], how="left")
    merged["booked_min"]  = merged["booked_min"].fillna(0)
    merged["idle_min"]    = (total_available_min - merged["booked_min"]).clip(lower=0)
    merged["utilization"] = (merged["booked_min"] / total_available_min).round(4)

    # attach room metadata
    merged = merged.merge(ROOMS[["room_id","room_name","room_type","hourly_cost"]], on="room_id")

    return merged

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def generate_all_data() -> dict[str, pd.DataFrame]:
    """
    Returns a dictionary of all DataFrames produced in Phase 1.
    Keys: 'rooms', 'providers', 'services', 'appointments', 'idle_time'
    """
    print("⚙️  Generating synthetic med spa operations data...")
    appointments = generate_appointments()
    idle_time    = compute_idle_time(appointments)

    datasets = {
        "rooms":        ROOMS,
        "providers":    PROVIDERS,
        "services":     SERVICES,
        "appointments": appointments,
        "idle_time":    idle_time,
    }

    # ── Quick summary ──────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  PHASE 1 — DATA GENERATION SUMMARY")
    print(f"{'─'*55}")
    print(f"  Simulation period : {START_DATE.date()}  →  "
          f"{(START_DATE + timedelta(days=SIM_DAYS-1)).date()}")
    print(f"  Rooms             : {len(ROOMS)}")
    print(f"  Providers         : {len(PROVIDERS)}")
    print(f"  Services          : {len(SERVICES)}")
    print(f"  Total appointments: {len(appointments):,}")
    print(f"  Completed         : {(appointments['status']=='Completed').sum():,}")
    print(f"  No-Shows          : {(appointments['status']=='No-Show').sum():,}")
    print(f"  Late Cancels      : {(appointments['status']=='Late Cancel').sum():,}")
    print(f"  Total Revenue     : ${appointments['actual_revenue'].sum():,.2f}")
    print(f"  Avg Daily Revenue : ${appointments.groupby('date')['actual_revenue'].sum().mean():,.2f}")
    print(f"  Idle time rows    : {len(idle_time):,}")
    print(f"  Avg Room Util.    : {idle_time['utilization'].mean():.1%}")
    print(f"{'─'*55}\n")

    return datasets


if __name__ == "__main__":
    data = generate_all_data()

    # Preview each DataFrame
    for name, df in data.items():
        print(f"\n{'═'*55}")
        print(f"  DataFrame: {name.upper()}  ({df.shape[0]} rows × {df.shape[1]} cols)")
        print(f"{'═'*55}")
        print(df.head(4).to_string(index=False))
