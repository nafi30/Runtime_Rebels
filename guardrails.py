from typing import List, Dict, Any
from schemas import DirectiveType, DirectiveInterpretation

def validate_and_sanitize_directives(
    raw_directives: List[Dict[str, Any]], 
    operator_notes: List[str], 
    battery_capacity: float
) -> List[DirectiveInterpretation]:
    """
    Deterministic validator enforcing canonical GridWise rules:
    1. Returns exactly one entry per note in strict note_index sequence (0..N-1).
    2. Applies is False only for no_op (with structured_adjustment = None).
    3. Applies is True for all other supported directives.
    4. Hours must be unique integers 0..23 in ascending order.
    5. Clamps factors (0.0..1.0) and reserves (0.0..capacity_kwh).
    """
    sanitized: List[DirectiveInterpretation] = []
    
    # Map raw directives by note_index for safe lookup
    lookup = {}
    if isinstance(raw_directives, list):
        for item in raw_directives:
            if isinstance(item, dict) and "note_index" in item:
                try:
                    lookup[int(item["note_index"])] = item
                except (ValueError, TypeError):
                    continue

    for idx, note in enumerate(operator_notes):
        raw = lookup.get(idx, {})
        
        dtype_str = str(raw.get("directive_type", "no_op")).strip()
        explanation = str(raw.get("explanation", "")).strip()
        if not explanation:
            explanation = "Directive parsed and validated." if dtype_str != "no_op" else "This note does not affect today's 24-hour energy schedule."

        # Validate Directive Type
        if dtype_str not in [e.value for e in DirectiveType]:
            dtype = DirectiveType.no_op
        else:
            dtype = DirectiveType(dtype_str)

        # Force no_op contract
        if dtype == DirectiveType.no_op or not raw.get("applies", True):
            sanitized.append(DirectiveInterpretation(
                note_index=idx,
                applies=False,
                directive_type=DirectiveType.no_op,
                structured_adjustment=None,
                explanation=explanation
            ))
            continue

        # Non-no_op directives MUST have applies = True
        adj = raw.get("structured_adjustment")
        if not isinstance(adj, dict):
            # Fallback to no_op if adjustment payload is missing
            sanitized.append(DirectiveInterpretation(
                note_index=idx,
                applies=False,
                directive_type=DirectiveType.no_op,
                structured_adjustment=None,
                explanation="Malformed adjustment data; defaulted to no_op."
            ))
            continue

        # Sanitize hours: unique integers, 0..23, ascending order
        raw_hours = adj.get("hours", [])
        clean_hours = []
        if isinstance(raw_hours, list):
            for h in raw_hours:
                try:
                    h_int = int(h)
                    if 0 <= h_int <= 23:
                        clean_hours.append(h_int)
                except (ValueError, TypeError):
                    continue
        clean_hours = sorted(list(set(clean_hours)))
        
        if not clean_hours:
            sanitized.append(DirectiveInterpretation(
                note_index=idx,
                applies=False,
                directive_type=DirectiveType.no_op,
                structured_adjustment=None,
                explanation="No valid hours identified; defaulted to no_op."
            ))
            continue

        clean_adj: Dict[str, Any] = {"hours": clean_hours}

        if dtype == DirectiveType.solar_reduction:
            try:
                factor = float(adj.get("factor", 1.0))
                clean_adj["factor"] = round(max(0.0, min(1.0, factor)), 4)
            except (ValueError, TypeError):
                clean_adj["factor"] = 1.0

        elif dtype == DirectiveType.minimum_battery_reserve:
            try:
                res_val = float(adj.get("minimum_energy_kwh", 0.0))
                # Handle percentage if erroneously passed as fraction
                if 0.0 < res_val <= 1.0 and "percent" in note.lower():
                    res_val = res_val * battery_capacity
                clean_adj["minimum_energy_kwh"] = round(max(0.0, min(battery_capacity, res_val)), 2)
            except (ValueError, TypeError):
                clean_adj["minimum_energy_kwh"] = 0.0

        elif dtype == DirectiveType.max_grid_window:
            try:
                grid_cap = float(adj.get("max_grid_kwh", 0.0))
                clean_adj["max_grid_kwh"] = round(max(0.0, grid_cap), 2)
            except (ValueError, TypeError):
                clean_adj["max_grid_kwh"] = 0.0

        sanitized.append(DirectiveInterpretation(
            note_index=idx,
            applies=True,
            directive_type=dtype,
            structured_adjustment=clean_adj,
            explanation=explanation
        ))

    return sanitized
