import os
import sys
import os.path as pth
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Set page layout
st.set_page_config(layout="wide", page_title="Supply Chain Dashboard")

# Configure path so we can import fastuav core logic
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if not getattr(sys, 'frozen', False):
    src_path = pth.abspath(pth.join(current_dir, '..', '..'))
    if src_path not in sys.path:
        sys.path.append(src_path)

try:
    from fastuav.models.supply_chain.model import run_supply_chain_scenario
except ImportError:
    st.error("Could not import fastuav.models.supply_chain.model. Please check your PYTHONPATH.")
    st.stop()

# ---------------- Load Baseline Data ----------------
SOURCE_FOLDER_PATH = pth.join(current_dir, 'data', 'source_files')

@st.cache_data
def load_data():
    catalog_df = pd.read_csv(pth.join(SOURCE_FOLDER_PATH, 'supplier_parts_catalog_sample.csv'))
    map_df = pd.read_csv(pth.join(SOURCE_FOLDER_PATH, 'bom_to_part_family_map_sample.csv'))
    assump_df = pd.read_csv(pth.join(SOURCE_FOLDER_PATH, 'cost_leadtime_assumptions_sample.csv'))
    bom_df = pd.read_csv(pth.join(SOURCE_FOLDER_PATH, 'supply_chain_bom_quadcopter_template.csv'))
    return catalog_df, map_df, assump_df, bom_df

try:
    catalog_df_base, map_df_base, assump_df_base, bom_df_base = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

assump_base = dict(zip(assump_df_base['key'], assump_df_base['value']))
critical_path_mode_val = str(assump_base.get('critical_path_mode', 'max')).strip().lower()

comp_ids = bom_df_base['component_id'].dropna().tolist()

# ---------------- UI Setup ----------------
st.title("Interactive Supply Chain Dashboard")

# Sidebar - Global Controls
st.sidebar.header("Global Controls")
w_strategy = st.sidebar.selectbox('Strategy', options=['min_cost', 'min_lead_time', 'min_risk'], index=0)
w_perf_margin = st.sidebar.slider('Perf Margin', 0.7, 1.3, 1.0, 0.01)
w_use_cont = st.sidebar.checkbox('Use Continuous Model', value=True)

st.sidebar.divider()
w_quality = st.sidebar.slider('Quality Buffer Ratio', 0.0, 0.3, 0.05, 0.01)
w_scrap = st.sidebar.slider('Scrap Ratio', 0.0, 0.3, 0.03, 0.01)
w_risk_mult = st.sidebar.slider('Risk Cost Multiplier', 0.0, 0.8, 0.15, 0.01)
w_logistics = st.sidebar.slider('Logistics Buffer (Days)', 0, 45, 7, 1)

# Sidebar - Component Overrides
st.sidebar.header("Component Cost & Lead Time Adjustments")
overrides = {}
for c in comp_ids:
    st.sidebar.markdown(f"**{c}**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        cost_mult = st.number_input(f"Cost Mult.", min_value=0.5, max_value=2.0, value=1.0, step=0.1, key=f"cost_{c}")
    with col2:
        lead_delta = st.number_input(f"Lead Δ (Days)", min_value=-30, max_value=60, value=0, step=5, key=f"lead_{c}")
    overrides[c] = {'cost_mult': cost_mult, 'lead_delta_days': lead_delta}


# ---------------- Logic Execution ----------------
df, summ, mat_df = run_supply_chain_scenario(
    bom_df=bom_df_base,
    catalog_df=catalog_df_base,
    map_df=map_df_base,
    quality_buffer_ratio=w_quality,
    scrap_ratio=w_scrap,
    risk_cost_multiplier=w_risk_mult,
    logistics_buffer_days=w_logistics,
    performance_margin_factor=w_perf_margin,
    selection_strategy=w_strategy,
    use_continuous_model=w_use_cont,
    component_overrides=overrides,
    critical_path_mode=critical_path_mode_val
)

ok_df = df[df['selection_status'].isin(['catalog_selected', 'modeled_continuous'])].copy()

# ---------------- Plotting ----------------
fig = make_subplots(
    rows=3, cols=2,
    specs=[[{'type': 'xy'}, {'type': 'xy'}], 
           [{'type': 'domain'}, {'type': 'domain'}],
           [{'type': 'table', 'colspan': 2}, None]],
    subplot_titles=("Cost Breakdown (Adjusted)", "Lead Time", "Cost Share", "Raw Material Breakdown (Mass)", "Summary"),
    vertical_spacing=0.10,
    row_heights=[0.33, 0.33, 0.34]
)
fig.update_layout(height=900, margin=dict(t=50, b=50, l=30, r=30), showlegend=False)

if not ok_df.empty:
    # 1. Cost Bar
    fig.add_trace(go.Bar(
        x=ok_df['component_id'], 
        y=ok_df['line_cost_risk_adjusted_usd'], 
        name='Cost',
        marker_color=np.where(ok_df['selection_status']=='modeled_continuous', 'orange', 'royalblue')
    ), row=1, col=1)
    
    # 2. Lead Bar
    fig.add_trace(go.Bar(
        x=ok_df['component_id'], 
        y=ok_df['lead_time_days'], 
        name='Lead',
        marker_color='green'
    ), row=1, col=2)
    
    # 3. Pie/Donut Share
    fig.add_trace(go.Pie(
        labels=ok_df['component_id'], 
        values=ok_df['line_cost_usd'], 
        name='Share', 
        hole=0.4
    ), row=2, col=1)
    
    # 4. Summary Table
    fig.add_trace(go.Table(
        header=dict(values=['Metric', 'Value']),
        cells=dict(values=[summ['metric'], summ['value'].round(1)])
    ), row=3, col=1)

if not mat_df.empty:
    # 5. Sunburst Materials
    ids = ['Current Design']
    labels = ['Current Design']
    parents = ['']
    values = [mat_df['mass_kg'].sum()]
    
    comp_grp = mat_df.groupby('component_id')['mass_kg'].sum()
    ids += comp_grp.index.tolist()
    labels += comp_grp.index.tolist()
    parents += ['Current Design'] * len(comp_grp)
    values += comp_grp.values.tolist()
    
    for _, row in mat_df.iterrows():
        cid = row['component_id']
        mat = row['material']
        uid = f"{cid} - {mat}"
        ids.append(uid)
        labels.append(mat)
        parents.append(cid)
        values.append(row['mass_kg'])
        
    fig.add_trace(go.Sunburst(
        ids=ids, labels=labels, parents=parents, values=values, branchvalues="total", name='Materials'
    ), row=2, col=2)


st.plotly_chart(fig, use_container_width=True)

