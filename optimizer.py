from typing import List, Dict, Optional
import numpy as np
from scipy.optimize import linprog

from schemas import (
    OptimizeRequest,
    DirectiveInterpretation,
    DirectiveType,
    BatteryAction,
    HourlyPlanEntry,
    OptimizeResponse,
)


def solve_24h_schedule(
    request: OptimizeRequest,
    directives: List[DirectiveInterpretation],
) -> OptimizeResponse:
    """
    Solves the 24-hour campus energy schedule using SciPy HiGHS linear programming.

    120 Decision variables (5 per hour for hours t = 0..23):
      - x[5*t + 0]: grid[t]
      - x[5*t + 1]: solar_used[t]
      - x[5*t + 2]: charge[t]
      - x[5*t + 3]: discharge[t]
      - x[5*t + 4]: E[t] (battery energy stored after hour t)
    """
    # 1. Organize hourly data in strict 0..23 order
    hours_map = {h.hour: h for h in request.hours}
    if len(hours_map) != 24 or any(t not in hours_map for t in range(24)):
        raise ValueError("OptimizeRequest must contain complete 24-hour data (hours 0 to 23).")
    hour_data = [hours_map[t] for t in range(24)]

    battery = request.battery

    # 2. Parse directives into per-hour bounds / restrictions
    effective_solar_factor: List[float] = [1.0] * 24
    directive_min_reserve: List[float] = [0.0] * 24
    allow_charge: List[bool] = [True] * 24
    allow_discharge: List[bool] = [True] * 24
    max_grid_limit: List[Optional[float]] = [None] * 24

    for d in directives:
        if not d.applies or not d.structured_adjustment:
            continue

        adj = d.structured_adjustment
        hours = adj.get("hours", [])
        if not isinstance(hours, list):
            continue

        if d.directive_type == DirectiveType.solar_reduction:
            factor = float(adj.get("factor", 1.0))
            for t in hours:
                if 0 <= t <= 23:
                    effective_solar_factor[t] = min(effective_solar_factor[t], factor)

        elif d.directive_type == DirectiveType.minimum_battery_reserve:
            min_kwh = float(adj.get("minimum_energy_kwh", 0.0))
            for t in hours:
                if 0 <= t <= 23:
                    directive_min_reserve[t] = max(directive_min_reserve[t], min_kwh)

        elif d.directive_type == DirectiveType.no_charge_window:
            for t in hours:
                if 0 <= t <= 23:
                    allow_charge[t] = False

        elif d.directive_type == DirectiveType.no_discharge_window:
            for t in hours:
                if 0 <= t <= 23:
                    allow_discharge[t] = False

        elif d.directive_type == DirectiveType.max_grid_window:
            grid_cap = float(adj.get("max_grid_kwh", 0.0))
            for t in hours:
                if 0 <= t <= 23:
                    if max_grid_limit[t] is None:
                        max_grid_limit[t] = grid_cap
                    else:
                        max_grid_limit[t] = min(max_grid_limit[t], grid_cap)

    # 3. Construct bounds for all 120 variables
    bounds: List[tuple] = [None] * 120
    for t in range(24):
        # grid[t]: [0, max_grid_kwh] if max_grid_window applies, else [0, None]
        bounds[5 * t + 0] = (0.0, max_grid_limit[t])

        # solar_used[t]: [0, effective_solar[t]]
        eff_solar = max(0.0, hour_data[t].solar_kwh * effective_solar_factor[t])
        bounds[5 * t + 1] = (0.0, eff_solar)

        # charge[t]: [0, max_charge_kwh_per_hour] (0.0 if no_charge_window)
        ub_charge = battery.max_charge_kwh_per_hour if allow_charge[t] else 0.0
        bounds[5 * t + 2] = (0.0, ub_charge)

        # discharge[t]: [0, max_discharge_kwh_per_hour] (0.0 if no_discharge_window)
        ub_discharge = battery.max_discharge_kwh_per_hour if allow_discharge[t] else 0.0
        bounds[5 * t + 3] = (0.0, ub_discharge)

        # E[t]: [effective_min_reserve[t], capacity_kwh]
        eff_min_res = max(battery.minimum_energy_kwh, directive_min_reserve[t])
        eff_min_res = min(eff_min_res, battery.capacity_kwh)
        bounds[5 * t + 4] = (eff_min_res, battery.capacity_kwh)

    # 4. Construct equality constraints: A_eq @ x == b_eq
    # Total equality rows = 24 (energy balance) + 24 (battery transition) + 1 (end-of-day neutrality) = 49
    A_eq = np.zeros((49, 120), dtype=float)
    b_eq = np.zeros(49, dtype=float)

    # Constraint 1: Hourly Energy Balance (rows 0..23)
    # grid[t] + solar_used[t] + discharge[t] - charge[t] == demand_kwh[t]
    for t in range(24):
        row = t
        A_eq[row, 5 * t + 0] = 1.0   # grid[t]
        A_eq[row, 5 * t + 1] = 1.0   # solar_used[t]
        A_eq[row, 5 * t + 3] = 1.0   # discharge[t]
        A_eq[row, 5 * t + 2] = -1.0  # charge[t]
        b_eq[row] = hour_data[t].demand_kwh

    # Constraint 2: Battery Energy Transition (rows 24..47)
    # t = 0: E[0] - charge[0] + discharge[0] == initial_energy_kwh
    # t > 0: E[t] - E[t-1] - charge[t] + discharge[t] == 0
    for t in range(24):
        row = 24 + t
        A_eq[row, 5 * t + 4] = 1.0   # E[t]
        A_eq[row, 5 * t + 2] = -1.0  # charge[t]
        A_eq[row, 5 * t + 3] = 1.0   # discharge[t]

        if t == 0:
            b_eq[row] = battery.initial_energy_kwh
        else:
            A_eq[row, 5 * (t - 1) + 4] = -1.0  # E[t-1]
            b_eq[row] = 0.0

    # Constraint 3: End-of-Day Neutrality (row 48)
    # E[23] == initial_energy_kwh
    A_eq[48, 5 * 23 + 4] = 1.0
    b_eq[48] = battery.initial_energy_kwh

    # 5. Objective Function: Minimize sum(grid[t] * tariff[t]) - 1e-5 * solar_used[t]
    c_obj = np.zeros(120, dtype=float)
    for t in range(24):
        c_obj[5 * t + 0] = hour_data[t].tariff_bdt_per_kwh
        c_obj[5 * t + 1] = -1e-5  # tiny preference to utilize available solar

    # 6. Solve via SciPy HiGHS LP Solver
    res = linprog(
        c=c_obj,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        raise ValueError(f"Linear programming optimization failed: {res.message}")

    x = res.x

    # 7. Post-Processing & Response Construction
    hourly_plan: List[HourlyPlanEntry] = []
    for t in range(24):
        grid_val = round(max(0.0, float(x[5 * t + 0])), 4)
        solar_val = round(max(0.0, float(x[5 * t + 1])), 4)
        charge_val = float(x[5 * t + 2])
        discharge_val = float(x[5 * t + 3])
        energy_val = round(max(0.0, float(x[5 * t + 4])), 4)

        if charge_val > 0.001:
            action = BatteryAction.charge
            b_kwh = round(charge_val, 4)
        elif discharge_val > 0.001:
            action = BatteryAction.discharge
            b_kwh = round(discharge_val, 4)
        else:
            action = BatteryAction.idle
            b_kwh = 0.0

        hourly_plan.append(
            HourlyPlanEntry(
                hour=t,
                grid_kwh=grid_val,
                solar_used_kwh=solar_val,
                battery_action=action,
                battery_kwh=b_kwh,
                battery_energy_after_kwh=energy_val,
            )
        )

    total_grid_kwh = round(sum(entry.grid_kwh for entry in hourly_plan), 2)
    total_cost_bdt = round(
        sum(entry.grid_kwh * hour_data[entry.hour].tariff_bdt_per_kwh for entry in hourly_plan),
        2,
    )
    peak_grid_kwh = round(max(entry.grid_kwh for entry in hourly_plan), 2)

    plan_summary = (
        f"Optimal 24-hour energy schedule generated successfully for scenario '{request.scenario_id}'. "
        f"Total grid import: {total_grid_kwh:.2f} kWh, Peak grid import: {peak_grid_kwh:.2f} kWh, "
        f"Total cost: {total_cost_bdt:.2f} BDT."
    )

    return OptimizeResponse(
        scenario_id=request.scenario_id,
        directive_interpretation=directives,
        hourly_plan=hourly_plan,
        total_grid_kwh=total_grid_kwh,
        total_cost_bdt=total_cost_bdt,
        peak_grid_kwh=peak_grid_kwh,
        plan_summary=plan_summary,
    )
