import pandas as pd
import numpy as np

from fastuav.models.supply_chain.motor.material_composition import (
    MotorMaterialComposition,
)

# ---------- Raw Material Model & Continuous Models ----------
# Mass fractions for the procurement (post-processing) layer.
#
# The MOTOR row is deliberately absent here: it is owned by the in-loop
# discipline model in ``supply_chain/motor/``. Keeping a second set of motor
# fractions in this file meant one motor could be priced two ways -- the two
# sets had drifted to copper 0.30 vs 0.25 and magnet 0.10 vs 0.12, and magnet
# is the dominant cost driver at ~85 USD/kg. Use ``motor_composition()``.
MATERIAL_COMPOSITION = {
    'battery_pack': {'Lithium': 0.03, 'Cobalt': 0.12, 'Nickel': 0.15, 'Graphite': 0.18, 'Copper': 0.11, 'Aluminum': 0.06, 'Electrolyte': 0.15, 'Plastic': 0.20},
    'esc': {'Silicon': 0.20, 'Copper': 0.30, 'PCB': 0.30, 'Plastic': 0.20},
    'propeller': {'Carbon Fiber': 0.40, 'Nylon/Plastic': 0.60},
    'frame_arms': {'Carbon Fiber': 0.70, 'Epoxy': 0.25, 'Aluminum': 0.05},
    'structure': {'Carbon Fiber': 0.60, 'Epoxy': 0.30, 'Aluminum': 0.10},
    'default': {'Misc': 1.0}
}

# Material names used by the in-loop motor model, mapped onto the labels this
# procurement layer reports on.
_MOTOR_MATERIAL_LABELS = {
    'f_copper': 'Copper',
    'f_magnet': 'Neodymium',
    'f_steel': 'Steel',
    'f_aluminum': 'Aluminum',
}


def motor_composition(**overrides):
    """
    Motor mass fractions, taken from the in-loop discipline model.

    Reads the declared defaults of :class:`MotorMaterialComposition` so the
    procurement layer and the MDAO discipline can never disagree. Any option
    of that component (``f_copper``, ``f_magnet``, ``f_steel``, ``f_aluminum``)
    may be overridden per call, matching what was set in the model YAML.

    The named fractions do not sum to 1 (defaults total 0.92); the discipline
    model books the balance as an ``other`` output (insulation, epoxy,
    fasteners). The same balance is returned here as ``'Other'`` so that
    material mass still sums to the component mass -- dropping it would
    silently understate ~8% of motor mass in the BOM.

    :return: dict of {material label: mass fraction}
    """
    declared = MotorMaterialComposition().options
    composition = {}
    for opt, label in _MOTOR_MATERIAL_LABELS.items():
        frac = overrides[opt] if opt in overrides else declared[opt]
        composition[label] = float(frac)
    composition['Other'] = max(0.0, 1.0 - sum(composition.values()))
    return composition

def estimate_continuous_cost(family, perf_val, perf_key):
    """
    Estimate cost using a continuous function when no exact catalog match is found.
    """
    if pd.isna(perf_val): return 0.0
    val = float(perf_val)
    if family == 'battery': return 20.0 + 0.20 * val
    elif family == 'motor': return 40.0 + 60.0 * val
    elif family == 'propulsion' and 'esc' in str(perf_key).lower(): return 15.0 + 0.04 * val
    elif family == 'propulsion' and 'diameter' in str(perf_key).lower(): return 5.0 + 200.0 * (val**2)
    elif family == 'structure': return 10.0 + 40.0 * val
    return 10.0

def estimate_continuous_lead_time(family, cost_usd):
    """
    Estimate lead time based on cost as a proxy for complexity.
    """
    return 7 + (cost_usd / 50.0)

def estimate_continuous_risk(family, perf_val):
    """
    Estimate supply risk score (0.0 - 1.0).
    """
    return 0.3

def run_supply_chain_scenario(
    bom_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    map_df: pd.DataFrame,
    quality_buffer_ratio: float,
    scrap_ratio: float,
    risk_cost_multiplier: float,
    logistics_buffer_days: int,
    performance_margin_factor: float,
    selection_strategy: str,
    use_continuous_model: bool,
    component_overrides: dict,
    critical_path_mode: str = 'max'
):
    """
    Run a single supply chain evaluation scenario.

    Parameters
    ----------
    bom_df : pd.DataFrame
        Bill of Materials with 'component_id', 'part_family', 'required_perf_key', 'required_perf_min', 'quantity', etc.
    catalog_df : pd.DataFrame
        Supplier parts catalog.
    map_df : pd.DataFrame
        Mapping from BOM component_id to part_family.
    quality_buffer_ratio : float
        Additional cost buffer for quality control (0.0 to 1.0).
    scrap_ratio : float
        Additional cost buffer for scrap rate (0.0 to 1.0).
    risk_cost_multiplier : float
        Multiplier for risk-adjusted cost.
    logistics_buffer_days : int
        Days added to lead time for logistics.
    performance_margin_factor : float
        Factor to adjust required performance (e.g. 1.1 means 10% higher performance required).
    selection_strategy : str
        'min_cost', 'min_lead_time', or 'min_risk'.
    use_continuous_model : bool
        If True, use continuous estimation functions when no catalog match is found.
    component_overrides : dict
        Dictionary of {component_id: {'cost_mult': float, 'lead_delta_days': float}}
    critical_path_mode : str, optional
        How to aggregate lead time: 'max' (parallel) or 'sum' (sequential). Default is 'max'.

    Returns
    -------
    selection_df : pd.DataFrame
        Detailed selection results per component.
    summary_df : pd.DataFrame
        Aggregated metrics (total cost, total lead time, etc.).
    material_df : pd.DataFrame
        Raw material breakdown.
    """
    
    # Merge BOM with Mapping
    # Logic note: original code did bom_df_base.merge(map_df_base...)
    # We should support 'part_family' being already in bom_df or coming from map_df
    calc_df = bom_df.copy()
    if 'part_family' not in calc_df.columns and not map_df.empty:
        calc_df = calc_df.merge(map_df, on='component_id', how='left')
    elif not map_df.empty:
        # If map exists, we might want to ensure we have the mapping even if columns exist
        # But simpler is to assumes caller provides map_df if needed
        # The original code did a left merge. Let's replicate that behavior safely.
        # Check if columns overlap to avoid _x _y
        cols_to_use = map_df.columns.difference(calc_df.columns).tolist()
        if 'component_id' not in cols_to_use:
             cols_to_use.append('component_id')
        if len(cols_to_use) > 1:
             calc_df = calc_df.merge(map_df[cols_to_use], on='component_id', how='left')

    selected_rows = []
    material_rows = [] # For storing raw material breakdown

    for _, row in calc_df.iterrows():
        comp_id = row.get('component_id', '')
        family = row.get('part_family', None)
        req_key = row.get('required_perf_key', None)
        req_min = row.get('required_perf_min', None)

        qty = row.get('quantity', 1)
        if pd.isna(qty): qty = row.get('default_quantity', 1)

        # Filter Catalog
        cands = pd.DataFrame()
        if family and not catalog_df.empty:
             if 'part_family' in catalog_df.columns:
                 cands = catalog_df[catalog_df['part_family'] == family].copy()

        effective_req_min = 0.0
        if pd.notna(req_min):
            effective_req_min = float(req_min) * float(performance_margin_factor)

        if not cands.empty and pd.notna(req_key) and pd.notna(req_min):
            if 'perf_key' in cands.columns and 'perf_value' in cands.columns:
                cands = cands[cands['perf_key'] == req_key]
                cands = cands[cands['perf_value'] >= effective_req_min]

        match_found = not cands.empty
        sel_part_num, sel_supplier, sel_unit_cost, sel_lead_time, sel_risk = None, None, 0.0, 0.0, 0.0
        status = 'no_feasible_supplier'
        
        if match_found:
            # Sort candidates based on strategy
            if selection_strategy == 'min_lead_time':
                best = cands.sort_values(by=['lead_time_days', 'unit_cost_usd'], ascending=[True, True]).iloc[0]
            elif selection_strategy == 'min_risk':
                best = cands.sort_values(by=['risk_score', 'unit_cost_usd'], ascending=[True, True]).iloc[0]
            else:
                best = cands.sort_values(by=['unit_cost_usd', 'lead_time_days'], ascending=[True, True]).iloc[0]
            
            sel_part_num, sel_supplier = best['part_number'], best['supplier']
            sel_unit_cost = float(best['unit_cost_usd'])
            sel_lead_time = float(best['lead_time_days']) if pd.notna(best['lead_time_days']) else 0.0
            sel_risk = float(best['risk_score']) if pd.notna(best['risk_score']) else 0.0
            status = 'catalog_selected'
            
        elif use_continuous_model and pd.notna(req_key) and pd.notna(req_min):
            sel_unit_cost = estimate_continuous_cost(family, effective_req_min, req_key)
            sel_lead_time = estimate_continuous_lead_time(family, sel_unit_cost)
            sel_risk = estimate_continuous_risk(family, effective_req_min)
            sel_part_num, sel_supplier = f"MODEL-EST-{family}", "Continuous_Model_Est"
            status = 'modeled_continuous'
        
        # Calculate Raw Material Mass
        mass = float(row.get('total_mass_kg', 0.0))
        if pd.isna(mass) or mass == 0:
             u_mass = float(row.get('unit_mass_kg', 0.0))
             if pd.notna(u_mass) and u_mass > 0: mass = u_mass * float(qty)
        
        if mass > 0:
            if comp_id == 'motor' or family == 'motor':
                # Motor is owned by the in-loop discipline model
                composition = motor_composition()
            else:
                comp_key = comp_id if comp_id in MATERIAL_COMPOSITION else (family if family in MATERIAL_COMPOSITION else 'default')
                composition = MATERIAL_COMPOSITION.get(comp_key, MATERIAL_COMPOSITION['default'])
            for mat, frac in composition.items():
                material_rows.append({
                    'component_id': comp_id,
                    'material': mat,
                    'mass_kg': mass * frac
                })

        if status == 'no_feasible_supplier':
             selected_rows.append({'component_id': comp_id,'selection_status': status, 'line_cost_risk_adjusted_usd': 0.0, 'line_cost_usd': 0.0, 'lead_time_days': 0.0})
             continue

        ov = component_overrides.get(comp_id, {})
        cost_mult = float(ov.get('cost_mult', 1.0))
        lead_delta = float(ov.get('lead_delta_days', 0.0))

        final_unit_cost = sel_unit_cost * cost_mult
        final_lead_time = max(0.0, sel_lead_time + lead_delta)
        line_cost =float(qty) * final_unit_cost
        line_cost = line_cost * (1.0 + float(quality_buffer_ratio) + float(scrap_ratio))
        line_cost_risk = line_cost * (1.0 + float(risk_cost_multiplier) * sel_risk)

        selected_rows.append({
            'component_id': comp_id, 'part_family': family, 'quantity': float(qty),
            'selected_part_number': sel_part_num, 'selected_supplier': sel_supplier,
            'unit_cost_usd': final_unit_cost, 'lead_time_days': final_lead_time, 'risk_score': sel_risk,
            'line_cost_usd': line_cost, 'line_cost_risk_adjusted_usd': line_cost_risk,
            'selection_status': status
        })

    selection_df_local = pd.DataFrame(selected_rows)
    material_df_local = pd.DataFrame(material_rows)
    
    valid_lines = pd.DataFrame()
    if not selection_df_local.empty and 'selection_status' in selection_df_local.columns:
        valid_lines = selection_df_local[selection_df_local['selection_status'].isin(['catalog_selected', 'modeled_continuous'])].copy()

    total_cost = valid_lines['line_cost_usd'].sum() if not valid_lines.empty else 0.0
    total_cost_risk = valid_lines['line_cost_risk_adjusted_usd'].sum() if not valid_lines.empty else 0.0
    
    # Lead time aggregation
    base_lead = 0.0
    if not valid_lines.empty:
        if critical_path_mode == 'sum':
            base_lead = valid_lines['lead_time_days'].sum()
        else:
            base_lead = valid_lines['lead_time_days'].max()
            
    total_lead = base_lead + float(logistics_buffer_days)
    
    unmatched = 0
    if not selection_df_local.empty and 'selection_status' in selection_df_local.columns:
        unmatched = int((selection_df_local['selection_status'] == 'no_feasible_supplier').sum())

    summary_df_local = pd.DataFrame([
        {'metric': 'total_cost_usd', 'value': total_cost},
        {'metric': 'total_cost_risk_adjusted_usd', 'value': total_cost_risk},
        {'metric': 'total_lead_time_days', 'value': total_lead},
        {'metric': 'n_unmatched', 'value': unmatched},
    ])
    return selection_df_local, summary_df_local, material_df_local
