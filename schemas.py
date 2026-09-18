from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class DirectiveType(str, Enum):
    solar_reduction = "solar_reduction"
    minimum_battery_reserve = "minimum_battery_reserve"
    no_charge_window = "no_charge_window"
    no_discharge_window = "no_discharge_window"
    max_grid_window = "max_grid_window"
    no_op = "no_op"

class BatteryAction(str, Enum):
    charge = "charge"
    discharge = "discharge"
    idle = "idle"

# --- Request Models ---
class HourData(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    demand_kwh: float = Field(..., ge=0.0)
    solar_kwh: float = Field(..., ge=0.0)
    tariff_bdt_per_kwh: float = Field(..., ge=0.0)

class BatteryData(BaseModel):
    capacity_kwh: float = Field(..., gt=0.0)
    initial_energy_kwh: float = Field(..., ge=0.0)
    minimum_energy_kwh: float = Field(..., ge=0.0)
    max_charge_kwh_per_hour: float = Field(..., ge=0.0)
    max_discharge_kwh_per_hour: float = Field(..., ge=0.0)

class OptimizeRequest(BaseModel):
    scenario_id: str
    operator_notes: List[str]
    hours: List[HourData]
    battery: BatteryData

# --- Response Models ---
class DirectiveInterpretation(BaseModel):
    note_index: int
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: Optional[Dict[str, Any]] = None
    explanation: str

class HourlyPlanEntry(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: BatteryAction
    battery_kwh: float
    battery_energy_after_kwh: float

class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str
