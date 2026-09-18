"""
Comprehensive local test suite for GridWise energy scheduler.
Validates the official public sample cases and system guardrails.
"""

import json
import os
import sys
from fastapi.testclient import TestClient

from main import app
from schemas import OptimizeRequest, OptimizeResponse, DirectiveType, BatteryAction

client = TestClient(app)

SAMPLE_FILE = "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"


def test_health_endpoint():
    """Verify GET /health returns HTTP 200 with status ok."""
    print("\n[TEST 1] Testing GET /health...")
    resp = client.get("/health")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    payload = resp.json()
    assert payload == {"status": "ok"}, f"Unexpected health payload: {payload}"
    print("  -> GET /health passed successfully!")


def test_official_sample_cases():
    """Load and test all official cases from the public sample file."""
    print(f"\n[TEST 2] Testing official cases from {SAMPLE_FILE}...")
    assert os.path.exists(SAMPLE_FILE), f"Missing sample file: {SAMPLE_FILE}"

    with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("cases", [])
    assert len(cases) > 0, "No cases found in sample file."

    for case in cases:
        case_id = case.get("id", "UNKNOWN")
        case_input = case.get("input", {})
        print(f"\n  Running case: {case_id}")

        # 1. Validate input against Pydantic schema
        req_obj = OptimizeRequest(**case_input)
        assert req_obj.scenario_id == case_id

        # 2. Call POST /optimize-energy
        resp = client.post("/optimize-energy", json=case_input)
        assert resp.status_code == 200, f"Case {case_id} failed with {resp.status_code}: {resp.text}"

        res_json = resp.json()

        # 3. Validate response model
        opt_resp = OptimizeResponse(**res_json)
        assert opt_resp.scenario_id == case_id
        assert len(opt_resp.hourly_plan) == 24
        assert len(opt_resp.directive_interpretation) == len(req_obj.operator_notes)

        # 4. Check specific directives for SAMPLE-01
        if case_id == "SAMPLE-01":
            d0 = opt_resp.directive_interpretation[0]
            assert d0.note_index == 0
            assert d0.applies is True
            assert d0.directive_type == DirectiveType.solar_reduction
            assert d0.structured_adjustment is not None
            assert d0.structured_adjustment.get("hours") == [12, 13]
            assert abs(float(d0.structured_adjustment.get("factor", 0.0)) - 0.25) < 0.01

            d1 = opt_resp.directive_interpretation[1]
            assert d1.note_index == 1
            assert d1.applies is False
            assert d1.directive_type == DirectiveType.no_op
            assert d1.structured_adjustment is None

        # 5. Validate Hourly Constraints & Physics
        battery = req_obj.battery
        hours_map = {h.hour: h for h in req_obj.hours}
        prev_energy = battery.initial_energy_kwh

        calc_grid = 0.0
        calc_cost = 0.0
        calc_peak = 0.0

        for t in range(24):
            entry = opt_resp.hourly_plan[t]
            h_data = hours_map[t]

            assert entry.hour == t

            # Recalculations
            calc_grid += entry.grid_kwh
            calc_cost += entry.grid_kwh * h_data.tariff_bdt_per_kwh
            calc_peak = max(calc_peak, entry.grid_kwh)

            # Battery Energy bounds
            assert entry.battery_energy_after_kwh >= battery.minimum_energy_kwh - 0.01, (
                f"Hour {t}: Battery energy {entry.battery_energy_after_kwh} below minimum {battery.minimum_energy_kwh}"
            )
            assert entry.battery_energy_after_kwh <= battery.capacity_kwh + 0.01, (
                f"Hour {t}: Battery energy {entry.battery_energy_after_kwh} exceeds capacity {battery.capacity_kwh}"
            )

            # Energy Balance: grid + solar_used + discharge - charge == demand
            charge_kwh = entry.battery_kwh if entry.battery_action == BatteryAction.charge else 0.0
            discharge_kwh = entry.battery_kwh if entry.battery_action == BatteryAction.discharge else 0.0

            supplied = entry.grid_kwh + entry.solar_used_kwh + discharge_kwh - charge_kwh
            assert abs(supplied - h_data.demand_kwh) < 0.05, (
                f"Hour {t}: Energy balance failed. Supplied: {supplied:.4f}, Demand: {h_data.demand_kwh}"
            )

            # Battery Transition: E[t] == E[t-1] + charge - discharge
            expected_energy = prev_energy + charge_kwh - discharge_kwh
            assert abs(entry.battery_energy_after_kwh - expected_energy) < 0.05, (
                f"Hour {t}: Battery transition mismatch. E[t]={entry.battery_energy_after_kwh}, Expected={expected_energy}"
            )
            prev_energy = entry.battery_energy_after_kwh

        # 6. Check End-of-Day Battery Neutrality
        final_energy = opt_resp.hourly_plan[23].battery_energy_after_kwh
        assert abs(final_energy - battery.initial_energy_kwh) < 0.01, (
            f"End-of-day neutrality violated: final {final_energy} != initial {battery.initial_energy_kwh}"
        )

        # 7. Check Recalculated Totals Match within 0.05 tolerance
        assert abs(opt_resp.total_grid_kwh - calc_grid) <= 0.05, (
            f"total_grid_kwh mismatch: {opt_resp.total_grid_kwh} vs {calc_grid}"
        )
        assert abs(opt_resp.total_cost_bdt - calc_cost) <= 0.05, (
            f"total_cost_bdt mismatch: {opt_resp.total_cost_bdt} vs {calc_cost}"
        )
        assert abs(opt_resp.peak_grid_kwh - calc_peak) <= 0.05, (
            f"peak_grid_kwh mismatch: {opt_resp.peak_grid_kwh} vs {calc_peak}"
        )

        print(f"    Case {case_id} Metrics:")
        print(f"      - Total Grid Energy : {opt_resp.total_grid_kwh:.2f} kWh")
        print(f"      - Total Cost        : {opt_resp.total_cost_bdt:.2f} BDT")
        print(f"      - Peak Grid Load    : {opt_resp.peak_grid_kwh:.2f} kWh")
        print(f"      - End-of-Day Neutral: {final_energy:.2f} == {battery.initial_energy_kwh:.2f} kWh")
        print(f"  -> Case {case_id} PASSED all contract checks!")


def main():
    print("==================================================")
    print("  GRIDWISE LOCAL VALIDATION TEST SUITE           ")
    print("==================================================")
    try:
        test_health_endpoint()
        test_official_sample_cases()
        print("\n==================================================")
        print("  ALL SYSTEM CHECKS & TESTS PASSED (100% GREEN)  ")
        print("==================================================")
    except AssertionError as e:
        print(f"\n[FAILED] Assertion Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
