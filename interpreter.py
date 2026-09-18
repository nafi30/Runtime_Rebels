import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from openai import OpenAI

logger = logging.getLogger("gridwise.interpreter")

SYSTEM_PROMPT = """You are an expert energy scheduling assistant for the GridWise microgrid platform.
Your task is to analyze natural-language operator notes and extract structured operational directives for a 24-hour campus energy schedule.

For each note provided in the input, you must produce exactly one directive entry corresponding to that note's index (0, 1, ... N-1).

SUPPORTED DIRECTIVE TYPES:
1. "solar_reduction":
   - PV/solar output is reduced, degraded, shaded, or undergoing panel cleaning.
   - applies: true
   - structured_adjustment: {"hours": [int, ...], "factor": float}
   - CRITICAL: "factor" is the USABLE FRACTION REMAINING (0.0 to 1.0).
     * An 80% reduction means factor = 0.2.
     * A 30% reduction means factor = 0.7.
     * "Roughly 25% of forecast" means factor = 0.25.
     * Complete loss of solar means factor = 0.0.

2. "minimum_battery_reserve":
   - A minimum battery state-of-charge or reserve buffer is mandated.
   - applies: true
   - structured_adjustment: {"hours": [int, ...], "minimum_energy_kwh": float}
   - If expressed as a percentage (e.g., "50% of battery capacity"), multiply by battery_capacity_kwh: minimum_energy_kwh = (percentage / 100.0) * battery_capacity_kwh.
   - If expressed in kWh directly, use that value.

3. "no_charge_window":
   - The battery is strictly prohibited from charging (e.g., grid peak tariff hours, transformer maintenance).
   - applies: true
   - structured_adjustment: {"hours": [int, ...]}

4. "no_discharge_window":
   - The battery is strictly prohibited from discharging (e.g., conserving reserve for emergency, testing).
   - applies: true
   - structured_adjustment: {"hours": [int, ...]}

5. "max_grid_window":
   - A maximum limit or cap on imported grid power is enforced.
   - applies: true
   - structured_adjustment: {"hours": [int, ...], "max_grid_kwh": float}

6. "no_op":
   - The note is unrelated to electrical dispatch/energy constraints (e.g., cafeteria specials, classroom bookings, administrative announcements, maintenance that does not affect microgrid operations).
   - applies: false
   - directive_type: "no_op"
   - structured_adjustment: null

TIME WINDOW CONVENTION (START-INCLUSIVE, END-EXCLUSIVE):
- "1 PM to 3 PM" -> [13, 14]
- "noon until 2 PM" -> [12, 13]
- "6 PM until 9 PM" -> [18, 19, 20]
- "between 09:00 and 12:00" -> [9, 10, 11]
- All hours must be unique integers strictly in ascending order within [0, 23].

OUTPUT SPECIFICATION:
You MUST return a single valid JSON object with the exact key "directives":
{
  "directives": [
    {
      "note_index": 0,
      "directive_type": "solar_reduction",
      "applies": true,
      "structured_adjustment": {
        "hours": [13, 14],
        "factor": 0.2
      },
      "explanation": "Panel cleaning from 1 PM to 3 PM leaves 20% solar output remaining."
    }
  ]
}
"""


def _heuristic_fallback_extraction(
    operator_notes: List[str],
    battery_capacity: float,
) -> List[Dict[str, Any]]:
    """
    Heuristic rule-based fallback used when OpenAI API is unreachable or unconfigured.
    Guarantees deterministic safe operation during offline evaluation.
    """
    directives: List[Dict[str, Any]] = []

    def parse_hour_range(text: str) -> List[int]:
        t_lower = text.lower()
        m = re.search(
            r"(\d{1,2})(?::\d{2})?\s*(am|pm)?\s*(?:to|until|and|-)\s*(\d{1,2})(?::\d{2})?\s*(am|pm)?",
            t_lower,
        )
        if m:
            start_h = int(m.group(1))
            start_ampm = m.group(2)
            end_h = int(m.group(3))
            end_ampm = m.group(4)

            if end_ampm == "pm" and end_h < 12:
                end_h += 12
            elif end_ampm == "am" and end_h == 12:
                end_h = 0

            if start_ampm == "pm" and start_h < 12:
                start_h += 12
            elif start_ampm == "am" and start_h == 12:
                start_h = 0
            elif not start_ampm and end_ampm == "pm" and start_h < 12 and start_h < (end_h - 12):
                start_h += 12

            return [h for h in range(start_h, end_h) if 0 <= h <= 23]

        if "noon until 2 pm" in t_lower or "noon to 2 pm" in t_lower or "12 pm to 2 pm" in t_lower:
            return [12, 13]
        return []

    for idx, note in enumerate(operator_notes):
        n_lower = note.lower()
        hours = parse_hour_range(n_lower)

        # 1. Solar Reduction
        if any(k in n_lower for k in ["solar", "pv", "panel", "cloud", "shade", "cleaning"]):
            factor = 1.0
            m_pct = re.search(r"(\d+(?:\.\d+)?)\s*%", n_lower)
            if m_pct:
                val = float(m_pct.group(1)) / 100.0
                if any(w in n_lower for w in ["reduc", "drop", "loss", "down", "cut"]):
                    factor = max(0.0, 1.0 - val)
                else:
                    factor = val
            elif "roughly 25%" in n_lower:
                factor = 0.25

            if hours:
                directives.append({
                    "note_index": idx,
                    "directive_type": "solar_reduction",
                    "applies": True,
                    "structured_adjustment": {"hours": hours, "factor": round(factor, 4)},
                    "explanation": f"Heuristic extraction: solar reduction to factor {factor}."
                })
                continue

        # 2. No charge window
        if "charge" in n_lower and any(w in n_lower for w in ["no charge", "do not charge", "stop charging", "prevent charging", "restrict charging"]):
            if hours:
                directives.append({
                    "note_index": idx,
                    "directive_type": "no_charge_window",
                    "applies": True,
                    "structured_adjustment": {"hours": hours},
                    "explanation": "Heuristic extraction: no charge window."
                })
                continue

        # 3. No discharge window
        if "discharge" in n_lower and any(w in n_lower for w in ["no discharge", "do not discharge", "stop discharging", "prevent discharging"]):
            if hours:
                directives.append({
                    "note_index": idx,
                    "directive_type": "no_discharge_window",
                    "applies": True,
                    "structured_adjustment": {"hours": hours},
                    "explanation": "Heuristic extraction: no discharge window."
                })
                continue

        # 4. Minimum battery reserve
        if any(w in n_lower for w in ["reserve", "buffer", "minimum energy", "min battery"]):
            min_kwh = 0.0
            m_kwh = re.search(r"(\d+(?:\.\d+)?)\s*kwh", n_lower)
            if m_kwh:
                min_kwh = float(m_kwh.group(1))
            else:
                m_pct = re.search(r"(\d+(?:\.\d+)?)\s*%", n_lower)
                if m_pct:
                    min_kwh = (float(m_pct.group(1)) / 100.0) * battery_capacity

            if hours and min_kwh > 0:
                directives.append({
                    "note_index": idx,
                    "directive_type": "minimum_battery_reserve",
                    "applies": True,
                    "structured_adjustment": {"hours": hours, "minimum_energy_kwh": round(min_kwh, 2)},
                    "explanation": f"Heuristic extraction: minimum battery reserve of {min_kwh} kWh."
                })
                continue

        # 5. Max grid window
        if any(w in n_lower for w in ["grid limit", "grid cap", "max grid", "import limit"]):
            m_grid = re.search(r"(\d+(?:\.\d+)?)\s*kwh?", n_lower)
            if m_grid and hours:
                directives.append({
                    "note_index": idx,
                    "directive_type": "max_grid_window",
                    "applies": True,
                    "structured_adjustment": {"hours": hours, "max_grid_kwh": round(float(m_grid.group(1)), 2)},
                    "explanation": f"Heuristic extraction: max grid import limit of {m_grid.group(1)} kWh."
                })
                continue

        # 6. Default to no_op
        directives.append({
            "note_index": idx,
            "directive_type": "no_op",
            "applies": False,
            "structured_adjustment": None,
            "explanation": "Note does not affect 24-hour energy dispatch."
        })

    return directives


def extract_directives_llm(
    operator_notes: List[str],
    battery_capacity: float,
) -> List[Dict[str, Any]]:
    """
    Extracts structured operational directives from operator notes via LLM.
    Returns a list of raw dicts for validation and sanitization by guardrails.py.
    """
    if not operator_notes:
        return []

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment. Using heuristic fallback.")
        return _heuristic_fallback_extraction(operator_notes, battery_capacity)

    try:
        base_url = os.environ.get("OPENAI_BASE_URL")
        client = OpenAI(api_key=api_key, base_url=base_url if base_url else None)
        model = os.environ.get("LLM_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))

        notes_payload = [
            {"note_index": i, "note": note}
            for i, note in enumerate(operator_notes)
        ]

        user_content = (
            f"Battery Storage Capacity: {battery_capacity:.2f} kWh\n\n"
            f"Operator Notes to Parse:\n{json.dumps(notes_payload, indent=2)}\n\n"
            "Return the extracted directives JSON object with key 'directives'."
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        raw_content = response.choices[0].message.content or "{}"
        parsed_json = json.loads(raw_content)

        raw_directives = parsed_json.get("directives")
        if isinstance(raw_directives, list):
            return raw_directives

        if isinstance(parsed_json, list):
            return parsed_json

        logger.warning("LLM response did not contain a 'directives' list; using heuristic fallback.")
        return _heuristic_fallback_extraction(operator_notes, battery_capacity)

    except Exception as e:
        logger.error(f"Error calling LLM API for directive extraction: {e}. Using fallback.", exc_info=True)
        return _heuristic_fallback_extraction(operator_notes, battery_capacity)
