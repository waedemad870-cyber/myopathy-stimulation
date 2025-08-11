import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
from io import BytesIO
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(
    page_title=" Myopathy: Gene Expression & Muscle Health Simulatorr",
    page_icon="🧬",
    layout="wide"
)

# --- DARK THEME STYLE ---
st.markdown("""
<style>
body, .main {background: linear-gradient(135deg, #0d102b 0%, #181b25 50%, #0f1419 100%);}
h1, h2, h3, h4, h5, h6 {
    color: #a3c2fd !important;
    text-shadow: 0 0 10px rgba(163, 194, 253, 0.3);
}
.info-card {
    background: linear-gradient(145deg, #192041, #21264b);
    color: #FFD700;
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 0 15px #181b25;
    margin: 10px 0;
    border: 1.5px solid #a3c2fd;
}
.explain-popup {
    background: linear-gradient(145deg, #21264b, #0d102b);
    color: #FFD700;
    border-radius: 14px;
    padding: 18px;
    margin: 10px 0;
    box-shadow: 0 0 10px #FFD700;
    border: 2px solid #FFD700;
}
.stTabs [data-baseweb="tab"] {
    background: linear-gradient(145deg, #192041, #21264b) !important;
    color: #FFD700 !important;
    border: 1px solid #a3c2fd;
    border-radius: 8px 8px 0 0;
    margin-right: 2px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(145deg, #21264b, #2a3052) !important;
    color: #ff5050 !important;
    box-shadow: 0 0 15px rgba(255, 80, 80, 0.3);
}
.download-btn {
    background: linear-gradient(90deg, #21264b, #FFD700, #a3c2fd);
    color: #181b25;
    border-radius: 10px;
    border: none;
    padding: 12px 18px;
    font-size: 1.1em;
    font-weight: bold;
    margin: 10px 0;
    box-shadow: 0 0 10px #FFD700;
}
.periodic-table-cell {
    display: inline-block;
    padding: 6px 8px;
    margin: 2px;
    border-radius: 5px;
    background: #181b25;
    color: #FFD700;
    font-weight: bold;
    text-align: center;
    border: 1px solid #a3c2fd;
    font-size: 12px;
    width: 32px;
}
.stAlert, .stInfo {
    font-size: 1.1em;
}
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE FOR ROLES ---
if "user_type" not in st.session_state:
    st.session_state.user_type = None
if "founder_ok" not in st.session_state:
    st.session_state.founder_ok = False

# --- ROLE SELECTION & FOUNDER PASSWORD ---
if st.session_state.user_type is None:
    st.markdown("<h4 style='color:#FFD700;'>Who are you?</h4>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Patient"):
            st.session_state.user_type = "patient"
    with col2:
        if st.button("Doctor"):
            st.session_state.user_type = "doctor"
    with col3:
        if st.button("Researcher"):
            st.session_state.user_type = "researcher"
    with col4:
        if st.button("Founder"):
            st.session_state.user_type = "founder"
    st.stop()
elif st.session_state.user_type == "founder" and not st.session_state.founder_ok:
    pwd = st.text_input("Enter founder password:", type="password")
    if st.button("Submit Password"):
        if pwd.lower() == "promiseme":
            st.session_state.founder_ok = True
            st.success("Access granted.")
        else:
            st.error("Wrong password. Try again.")
    st.stop()
else:
    if st.session_state.user_type == "patient":
        st.markdown("<h2 style='color:#FFD700;text-align:center;'>DON'T GIVE UP</h2>", unsafe_allow_html=True)
    elif st.session_state.user_type == "doctor":
        st.markdown("<h2 style='color:#FFD700;text-align:center;'>Don't let them down!</h2>", unsafe_allow_html=True)
    elif st.session_state.user_type == "researcher":
        st.markdown("<h2 style='color:#FFD700;text-align:center;'>Welcome to the Gene Expression!</h2>", unsafe_allow_html=True)
    elif st.session_state.user_type == "founder":
        st.markdown("<h2 style='color:#FFD700;text-align:center;'>Welcome Founder!</h2>", unsafe_allow_html=True)

# --- MAIN HEADER ---
st.markdown("<h1 style='text-align:center;'>🧬 Collagen 6A3 Myopathy Simulator</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>Gene Expression · Muscle Health · Regeneration Modeling</h3>", unsafe_allow_html=True)
st.markdown("---")

# --- HOW TO USE EXPANDER ---
with st.expander("ℹ️ How to Use This App", expanded=True):
    st.markdown("""
    <div class="info-card">
    <b>Welcome!</b><br>
    <ul>
        <li>Start by selecting your role above.</li>
        <li>Use the sidebar to set simulation parameters.</li>
        <li>Navigate tabs for results, lab experiments, theory, and more.</li>
        <li>Try the Theory Simulation for a visual sequence of gene therapy.</li>
        <li>Export data or create a PDF report for your records.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("<h2 style='color:#FFD700;'>Simulation Controls</h2>", unsafe_allow_html=True)
transcription_rate = st.sidebar.slider("Transcription Rate (mRNA production)", 0.1, 20.0, 5.0, 0.1)
degradation_rate = st.sidebar.slider("Degradation Rate (mRNA breakdown)", 0.01, 1.0, 0.1, 0.01)
regeneration_rate = st.sidebar.slider("Regeneration Rate (muscle recovery)", 0.01, 1.0, 0.05, 0.01)
simulation_time = st.sidebar.slider("Simulation Time (hours)", 10, 100, 50, 1)
time_steps = st.sidebar.slider("Time Steps (resolution)", 100, 1000, 500, 10)

fiber_type = st.sidebar.selectbox("Muscle Fiber Type", ["Slow-Twitch (Type I)", "Fast-Twitch (Type II)", "Mixed"])
mutation_severity = st.sidebar.selectbox("Mutation Severity", ["Mild", "Moderate", "Severe"], index=1)
enable_gene_therapy = st.sidebar.checkbox("Enable Gene Therapy", value=False)
if enable_gene_therapy:
    therapy_start_time = st.sidebar.slider("Therapy Start Time (hours)", 0, simulation_time, int(simulation_time//2), 1)
    therapy_duration = st.sidebar.slider("Therapy Duration (hours)", 1, int(simulation_time//2), 10, 1)
    therapy_boost = st.sidebar.slider("Therapy Boost Factor", 1.0, 5.0, 2.0, 0.1)
else:
    therapy_start_time = therapy_duration = therapy_boost = 0

enable_damage = st.sidebar.checkbox("Simulate Damage Event", value=False)
if enable_damage:
    damage_time = st.sidebar.slider("Damage Event Time (hours)", 0, simulation_time, int(simulation_time*0.75), 1)
    damage_severity = st.sidebar.slider("Damage Severity", 0.01, 0.9, 0.3, 0.01)
else:
    damage_time = damage_severity = 0

st.sidebar.markdown("---")
st.sidebar.markdown("""
*How to run:*  
- Install dependencies: pip install streamlit numpy pandas plotly pillow  
- Run: streamlit run app.py
- Try Streamlit Cloud for easy deployment!
""")

# --- SIMULATION FUNCTIONS ---
def simulate_gene_expression(tr, dr, sim_time, steps):
    dt = sim_time / steps
    t = np.linspace(0, sim_time, steps)
    mRNA = np.zeros(steps)
    mRNA[0] = 1.0
    for i in range(1, steps):
        dmRNA = tr - dr * mRNA[i-1]
        mRNA[i] = max(mRNA[i-1] + dmRNA * dt, 0)
    return mRNA, t

def simulate_gene_therapy(mRNA, t, start_time, duration, boost_factor):
    mRNA_therapy = mRNA.copy()
    start_idx = np.searchsorted(t, start_time)
    end_idx = np.searchsorted(t, start_time + duration)
    for i in range(start_idx, min(end_idx, len(mRNA))):
        mRNA_therapy[i] *= boost_factor
    return mRNA_therapy

def simulate_muscle_health(mRNA, regen_rate, steps):
    health = np.ones(steps)
    regen = np.zeros(steps)
    for i in range(1, steps):
        if mRNA[i] < 1.0:
            loss = (1.0 - mRNA[i]) * 0.03
            health[i] = max(health[i-1] - loss, 0.0)
        else:
            health[i] = min(health[i-1] + 0.01, 1.0)
        if health[i] < 1.0:
            regen[i] = regen_rate * (1.0 - health[i])
            health[i] = min(health[i] + regen[i], 1.0)
    return health, regen

def simulate_damage_event(health, t, damage_time, damage_severity):
    health_damaged = health.copy()
    idx = np.searchsorted(t, damage_time)
    if idx < len(health):
        health_damaged[idx] *= (1.0 - damage_severity)
        for i in range(idx+1, len(health)):
            health_damaged[i] = min(health_damaged[i], health_damaged[i-1] + 0.02)
    return health_damaged

def simulate_protein(mRNA, t):
    protein = np.zeros_like(mRNA)
    protein[0] = 1.0
    for i in range(1, len(mRNA)):
        synthesis = mRNA[i] * 0.5
        degradation = protein[i-1] * 0.1
        protein[i] = max(protein[i-1] + (synthesis - degradation) * (t[1] - t[0]), 0)
    return protein

def get_summary_stats(health, t):
    min_health = np.min(health)
    max_health = np.max(health)
    recovery_idx = np.where(health >= 0.99)[0]
    recovery_time = t[recovery_idx[0]] if len(recovery_idx) else None
    return {"min_health": min_health, "max_health": max_health, "recovery_time": recovery_time}

def export_csv(data_dict):
    df = pd.DataFrame(data_dict)
    return df.to_csv(index=False)

def save_fig_as_image(fig):
    buf = BytesIO()
    fig.write_image(buf, format='png')
    buf.seek(0)
    return buf

def generate_pdf_report(data_dict):
    df = pd.DataFrame(data_dict)
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

# --- THEORY SIMULATION VISUALS ---
def plot_dna_helix(fade=0):
    t = np.linspace(0, 4 * np.pi, 100)
    x1 = np.cos(t)
    y1 = np.sin(t)
    x2 = np.cos(t + np.pi)
    y2 = np.sin(t + np.pi)
    z = t
    helix_color = f'rgba(163,194,253,{1-fade})'
    broken_color = f'rgba(255,80,80,{fade})'
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x1, y=y1, z=z, mode='lines', line=dict(color=helix_color, width=6)))
    fig.add_trace(go.Scatter3d(x=x2, y=y2, z=z, mode='lines', line=dict(color=helix_color, width=6)))
    if fade > 0.01:
        fig.add_trace(go.Scatter3d(x=x1, y=y1, z=z, mode='lines', line=dict(color=broken_color, width=6, dash='dot')))
        fig.add_trace(go.Scatter3d(x=x2, y=y2, z=z, mode='lines', line=dict(color=broken_color, width=6, dash='dot')))
    fig.update_layout(template="plotly_dark", plot_bgcolor="#181b25", paper_bgcolor="#181b25",
                      margin=dict(l=0, r=0, t=0, b=0), showlegend=False, scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
    return fig

def plot_muscle_boost(boost=1):
    t = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(t) * boost
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, line=dict(color="#FFD700", width=5)))
    fig.update_layout(template="plotly_dark", plot_bgcolor="#181b25", paper_bgcolor="#181b25",
                      font_color="#FFD700", height=300, margin=dict(l=0, r=0, t=0, b=0))
    return fig

def plot_regeneration_curve(progress=1):
    t = np.linspace(0, 1, 100)
    y = np.minimum(t, progress)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, line=dict(color="#00ff88", width=6)))
    fig.update_layout(template="plotly_dark", plot_bgcolor="#181b25", paper_bgcolor="#181b25",
                      font_color="#00ff88", height=300, margin=dict(l=0, r=0, t=0, b=0))
    return fig

# --- TABS ---
tab_sim, tab_explain, tab_guide, tab_lab, tab_theory, tab_scitheory, tab_chatg1 = st.tabs([
    "Simulation Results", "Scientific Concepts", "User Guide", "Lab & Experiments", "Theory Simulation", "Scientific Theory Page", "Chaty G 1"
])

# --- DOWNLOAD BUTTONS ---
st.markdown("<h4 style='color:#FFD700'>Download the App</h4>", unsafe_allow_html=True)
col_download1, col_download2, col_download3, col_download4 = st.columns(4)
with col_download1:
    st.markdown("<a href='https://apps.apple.com/' target='_blank'><button class='download-btn'>iPhone (App Store)</button></a>", unsafe_allow_html=True)
with col_download2:
    st.markdown("<a href='https://play.google.com/store' target='_blank'><button class='download-btn'>Android (Google Play)</button></a>", unsafe_allow_html=True)
with col_download3:
    st.markdown("<a href='https://www.microsoft.com/en-us/store/apps/windows' target='_blank'><button class='download-btn'>Windows</button></a>", unsafe_allow_html=True)
with col_download4:
    st.markdown("<a href='https://apps.apple.com/us/genre/mac/id39' target='_blank'><button class='download-btn'>Mac</button></a>", unsafe_allow_html=True)

# --- TAB: SIMULATION RESULTS ---
with tab_sim:
    st.markdown("## 🧪 Simulation Visualization")
    run_sim = st.button("🚀 Run Simulation", key="run_sim_btn")
    if run_sim or st.session_state.get("last_sim", False):
        st.session_state["last_sim"] = True
        severity = {"Mild": (0.9, 1.1), "Moderate": (0.7, 1.3), "Severe": (0.5, 1.6)}
        tr_mult, dr_mult = severity[mutation_severity]
        adj_tr = transcription_rate * tr_mult
        adj_dr = degradation_rate * dr_mult
        if fiber_type == "Slow-Twitch (Type I)":
            regen_mult = 1.2
        elif fiber_type == "Fast-Twitch (Type II)":
            regen_mult = 0.8
        else:
            regen_mult = 1.0
        adj_regen = regeneration_rate * regen_mult

        mRNA, t = simulate_gene_expression(adj_tr, adj_dr, simulation_time, time_steps)
        if enable_gene_therapy and therapy_duration > 0:
            mRNA = simulate_gene_therapy(mRNA, t, therapy_start_time, therapy_duration, therapy_boost)
        health, regen = simulate_muscle_health(mRNA, adj_regen, time_steps)
        if enable_damage and damage_severity > 0:
            health = simulate_damage_event(health, t, damage_time, damage_severity)
        protein = simulate_protein(mRNA, t)
        stats = get_summary_stats(health, t)

        st.markdown("<h5 style='color:#a3c2fd'>mRNA Concentration</h5>", unsafe_allow_html=True)
        fig_mrna = go.Figure(data=[go.Scatter(x=t, y=mRNA, line=dict(color="#a3c2fd", width=2), name="mRNA")])
        fig_mrna.update_layout(template="plotly_dark", plot_bgcolor="#181b25", paper_bgcolor="#181b25", font_color="#a3c2fd", height=350)
        fig_mrna.update_xaxes(title_text="Time (hours)")
        fig_mrna.update_yaxes(title_text="mRNA")
        st.plotly_chart(fig_mrna, use_container_width=True)
        st.download_button("Download mRNA Graph (PNG)", save_fig_as_image(fig_mrna), "mRNA_graph.png", "image/png")

        st.markdown("<h5 style='color:#FFD700'>Muscle Health</h5>", unsafe_allow_html=True)
        fig_health = go.Figure(data=[go.Scatter(x=t, y=health, line=dict(color="#FFD700", width=2), name="Health")])
        fig_health.update_layout(template="plotly_dark", plot_bgcolor="#181b25", paper_bgcolor="#181b25", font_color="#FFD700", height=350)
        fig_health.update_xaxes(title_text="Time (hours)")
        fig_health.update_yaxes(title_text="Health (0-1)")
        st.plotly_chart(fig_health, use_container_width=True)
        st.download_button("Download Health Graph (PNG)", save_fig_as_image(fig_health), "health_graph.png", "image/png")

        st.markdown("<h5 style='color:#ff5050'>Regeneration Effect</h5>", unsafe_allow_html=True)
        fig_regen = go.Figure(data=[go.Scatter(x=t, y=regen, line=dict(color="#ff5050", width=2, dash="dot"), name="Regeneration")])
        fig_regen.update_layout(template="plotly_dark", plot_bgcolor="#181b25", paper_bgcolor="#181b25", font_color="#ff5050", height=350)
        fig_regen.update_xaxes(title_text="Time (hours)")
        fig_regen.update_yaxes(title_text="Regeneration")
        st.plotly_chart(fig_regen, use_container_width=True)
        st.download_button("Download Regeneration Graph (PNG)", save_fig_as_image(fig_regen), "regen_graph.png", "image/png")

        st.markdown("<h5 style='color:#00ff88'>Collagen Protein</h5>", unsafe_allow_html=True)
        fig_protein = go.Figure(data=[go.Scatter(x=t, y=protein, line=dict(color="#00ff88", width=2), name="Protein")])
        fig_protein.update_layout(template="plotly_dark", plot_bgcolor="#181b25", paper_bgcolor="#181b25", font_color="#00ff88", height=350)
        fig_protein.update_xaxes(title_text="Time (hours)")
        fig_protein.update_yaxes(title_text="Protein Level")
        st.plotly_chart(fig_protein, use_container_width=True)
        st.download_button("Download Protein Graph (PNG)", save_fig_as_image(fig_protein), "protein_graph.png", "image/png")

        st.markdown(
            f"<div class='info-card'>"
            f"<b>Min Health:</b> {stats['min_health']:.3f} &nbsp;&nbsp;"
            f"<b>Max Health:</b> {stats['max_health']:.3f} &nbsp;&nbsp;"
            f"<b>Time to Full Regeneration:</b> {stats['recovery_time']:.1f} hours"
            f"</div>", unsafe_allow_html=True
        )

        data_dict = {
            "Time": t,
            "mRNA": mRNA,
            "Health": health,
            "Regeneration": regen,
            "Protein": protein
        }
        csv_data = export_csv(data_dict)
        st.download_button("Download Simulation CSV", csv_data, "simulation_results.csv", "text/csv")

        st.markdown("#### Export Data as PDF Report")
        pdf_buf = generate_pdf_report(data_dict)
        st.download_button("Download PDF Report (CSV placeholder)", pdf_buf, "simulation_report.csv", "text/csv")

# --- TAB: SCIENTIFIC CONCEPTS ---
with tab_explain:
    st.markdown("## 🔬 Scientific Concepts")
    st.markdown("""
    *Gene Expression and Collagen 6A3*
    - Collagen VI (COL6A3) is critical for muscle structure and repair.
    - Mutations can lead to Bethlem myopathy or Ullrich congenital muscular dystrophy.
    - mRNA levels approximate collagen production.

    *Muscle Health Modeling*
    - Health ranges from 0 (damaged) to 1 (healthy).
    - Low collagen leads to declining tissue health.
    - Regeneration helps recover health after injury.

    *Simulation Equations*
    d(mRNA)/dt = transcription_rate - degradation_rate * mRNA
    d(Health)/dt: Decreases if mRNA low, recovers with regeneration_rate

    Typical collagen mRNA half-life: ~10-20 hours. Muscle regeneration in mice: ~7-14 days. You can tune the rates above to match published literature!
    """)

    st.markdown("""
    *References:*  
    - Bönnemann, C.G. "The collagen VI-related myopathies." Handbook of Clinical Neurology (2011)  
    - Lampe, A.K., & Bushby, K.M. "Collagen VI related muscle disorders." Journal of Medical Genetics (2005)
    """)

# --- TAB: USER GUIDE ---
with tab_guide:
    st.markdown("## 📖 User Guide")
    st.markdown("""
    *How to Use This App:*
    1. Select your role above.
    2. Adjust parameters in the sidebar (transcription, degradation, regeneration, etc.)
    3. Click through tabs to explore simulation results, scientific background, and chatbot.
    4. Download your results as CSV, PNG, or PDF for further analysis.

    *Tips:*
    - Enable gene therapy or damage events for advanced modeling.
    - Use the chatbot for interactive help and navigation.

    *Deployment:*
    - Run locally: streamlit run app.py
    - Deploy to [Streamlit Cloud](https://streamlit.io/cloud)
    """)

# --- TAB: LAB & EXPERIMENTS ---
with tab_lab:
    st.markdown("## 🧪 Lab & Experiments")
    st.markdown("<h4 style='color:#FFD700'>Periodic Table</h4>", unsafe_allow_html=True)
    periodic_elements = [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
        "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr",
        "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "In", "Sn", "Sb", "Te", "I", "Xe",
        "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn",
        "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
        "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Fl", "Lv", "Ts", "Og"
    ]
    pt_html = ""
    for idx, el in enumerate(periodic_elements):
        pt_html += f"<span class='periodic-table-cell'>{el}</span>"
        if (idx+1) % 18 == 0:
            pt_html += "<br>"
    st.markdown(pt_html, unsafe_allow_html=True)
    st.markdown("Use the periodic table for chemistry experiments. Select an element for more info.")
    selected_element = st.selectbox("Choose an element to view details", periodic_elements)
    st.info(f"Element: {selected_element} - More chemical and experimental info coming soon.")

    st.markdown("### Visual Experiments")
    chem_compounds = ["Collagen VI", "ATP", "Glucose", "Ca2+", "Mg2+", "Na+", "K+", "O2", "CO2"]
    compound = st.selectbox("Choose a compound to visualize", chem_compounds)
    comp_data = np.random.rand(10)
    fig_comp = px.bar(x=range(10), y=comp_data, color=comp_data, color_continuous_scale="Blues")
    fig_comp.update_layout(template="plotly_dark", plot_bgcolor="#181b25", paper_bgcolor="#181b25")
    st.plotly_chart(fig_comp, use_container_width=True)
    st.markdown("Experiment with the compound concentration above.")

# --- TAB: THEORY SIMULATION ---
with tab_theory:
    st.markdown("## 🧬 Run Theory Simulation")
    theory_ready = st.button("Run Theory Simulation", key="run_theory_btn")
    if theory_ready:
        st.markdown("### Step 1: Destroy the Gene")
        fade_steps = np.linspace(0, 1, 10)
        for f in fade_steps:
            fig_dna = plot_dna_helix(fade=f)
            st.plotly_chart(fig_dna, use_container_width=True)
            time.sleep(0.07)
        st.markdown("The gene (DNA helix) fades and breaks (red), representing disabling the faulty component.")

        st.markdown("### Step 2: Boost the Muscles")
        boost_levels = np.linspace(1, 3, 10)
        for b in boost_levels:
            fig_muscle = plot_muscle_boost(boost=b)
            st.plotly_chart(fig_muscle, use_container_width=True)
            time.sleep(0.07)
        st.markdown("Muscle graph increases, representing the stimulation of regeneration.")

        st.markdown("### Step 3: Regenerate")
        progress_steps = np.linspace(0, 1, 20)
        for p in progress_steps:
            fig_regen = plot_regeneration_curve(progress=p)
            st.plotly_chart(fig_regen, use_container_width=True)
            time.sleep(0.07)
        st.markdown("Regeneration curve grows from 0% to 100% (green).")
        st.success("Theory simulation complete! You can run it again or ask Chaty G1 for more theory.")

# --- TAB: SCIENTIFIC THEORY PAGE ---
with tab_scitheory:
    st.markdown("## 🧬 Scientific Theory")
    st.markdown("""
    ### Collagen VI-related Myopathies (Scientific Theory)
    Collagen VI-related myopathies are caused by mutations in the COL6A1, COL6A2, or COL6A3 genes.  
    These mutations affect the collagen VI protein, which is a critical component of the extracellular matrix in skeletal muscles.  
    The defective protein results in weakened muscle structure, causing progressive muscle weakness and related symptoms.

    **Concept Summary:**  
    The therapeutic approach I am exploring focuses on targeting the defective protein or gene responsible for the disease, with the goal of stimulating regeneration and repair but instead of replacing every damaged muscle fiber directly, the strategy is to remove or disable the harmful components and boost natural regenerative processes.

    **Proposed Methodology:**  
    1. **Target Identification** – Identify the defective COL6A3 gene sequence or misfolded protein causing the disease.  
    2. **Disable the Faulty Component** – Use molecular tools such as CRISPR-Cas9, RNA interference, or protein degradation methods to neutralize or remove defective gene activity or protein.  
    3. **Stimulate Regeneration** – Promote surrounding healthy cells to produce correct collagen VI and rebuild muscle structure.  
    4. **Systemic Delivery** – Administer therapy through the bloodstream or targeted delivery, minimizing the need for multiple injections.  
    5. **Long-Term Recovery** – Focus on gradual regeneration rather than immediate results, allowing the body to progressively heal.

    **Scientific Rationale and Reasoning:**  
    The human body has a natural ability to regenerate cells and produce proteins necessary for tissue repair and by removing harmful mutated proteins or gene activity, the cellular environment improves, allowing healthier tissue to form.  
    This concept is similar to the approaches used in cancer treatment where diseased cells are selectively destroyed, but here the focus is on disabling defective proteins and stimulating healthy regeneration while gene therapy is well-known, this approach combines removal and regeneration to potentially enhance effectiveness.

    There is an acceptance that some healthy cells or proteins may be sacrificed during treatment; this is considered a reasonable trade-off if it results in overall long-term improvement or stabilization.  
    The process requires careful timing and delivery to balance safety and efficacy.
    """)
    if st.button("Show Visual Sequence for Theory", key="run_theory_scientific_btn"):
        st.markdown("### Step 1: Destroy the Gene")
        fade_steps = np.linspace(0, 1, 10)
        for f in fade_steps:
            fig_dna = plot_dna_helix(fade=f)
            st.plotly_chart(fig_dna, use_container_width=True)
            time.sleep(0.07)
        st.markdown("The gene (DNA helix) fades and breaks (red), representing disabling the faulty component.")

        st.markdown("### Step 2: Boost the Muscles")
        boost_levels = np.linspace(1, 3, 10)
        for b in boost_levels:
            fig_muscle = plot_muscle_boost(boost=b)
            st.plotly_chart(fig_muscle, use_container_width=True)
            time.sleep(0.07)
        st.markdown("Muscle graph increases, representing the stimulation of regeneration.")

        st.markdown("### Step 3: Regenerate")
        progress_steps = np.linspace(0, 1, 20)
        for p in progress_steps:
            fig_regen = plot_regeneration_curve(progress=p)
            st.plotly_chart(fig_regen, use_container_width=True)
            time.sleep(0.07)
        st.markdown("Regeneration curve grows from 0% to 100% (green).")

# --- TAB: CHATY G 1 ---
with tab_chatg1:
    st.markdown("<h1 style='color:#FFD700;text-align:center;'>🤖 Chaty G 1 - Control & Chat</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
    <b>Meet Chaty G 1:</b><br>
    I am Chaty G 1, created by Waad Naser (IT Year 2). I can chat, answer science, guide you through the app, and even control simulations.
    <br><br>
    <b>How can I help?</b> Type your question or command below!
    </div>
    """, unsafe_allow_html=True)

    if "chat_history_g1" not in st.session_state:
        st.session_state.chat_history_g1 = []

    chat_input_g1 = st.text_input("Type a message to Chaty G 1:", key="chat_input_g1")
    send_g1 = st.button("Send to Chaty G 1")
    # --- Chaty G 1 Logic ---
    if send_g1 and chat_input_g1:
        question = chat_input_g1.strip().lower()
        st.session_state.chat_history_g1.append({"role":"user", "text": chat_input_g1})
        # Controls
        reply = None
        if "who created you" in question or "waad" in question or "your creator" in question:
            reply = "I, Chaty G 1, was created by Waad Naser, IT Year 2. Let's keep improving together!"
        elif "run theory" in question or "show theory" in question:
            reply = "Running the Theory Simulation now... Switch to 'Theory Simulation' tab for visuals!"
            st.session_state["run_theory_from_chatg1"] = True
        elif "help" in question or "how to use" in question:
            reply = "Go to the User Guide tab for step-by-step help. Or ask me anything specific!"
        elif "export" in question or "report" in question:
            reply = "You can export results as CSV, PNG, or PDF report from Simulation Results tab!"
        elif "collagen" in question or "6a3" in question or "protein" in question:
            reply = "Collagen VI (especially COL6A3) is vital for muscle structure. Mutations cause myopathies. Treatments may involve gene editing or boosting regeneration."
        elif "gene" in question or "crisp" in question or "cas9" in question:
            reply = "CRISPR-Cas9 and RNAi are promising tools to disable faulty genes. See Scientific Theory page for full methodology."
        elif "regeneration" in question or "muscle" in question:
            reply = "Regeneration is modeled in the simulation. Adjust the rate in sidebar. See results in graphs!"
        elif "doctor" in question or "scientist" in question or "report" in question:
            reply = "Doctors/scientists can export all results, simulate therapies, and guide patients using this app."
        elif "hello" in question or "hi" in question:
            reply = "Hello! How can I assist you today?"
        elif "vision" in question or "future" in question:
            reply = "Your vision shapes this platform. Let's keep improving, buddy!"
        elif "clear" in question:
            st.session_state.chat_history_g1 = []
            reply = "Chat cleared!"
        elif "experiment" in question or "lab" in question:
            reply = "Go to the Lab & Experiments tab for chemistry, periodic table, and compound visualization!"
        elif "periodic" in question or "table" in question:
            reply = "Periodic table is on the Lab & Experiments tab. Click elements for info."
        else:
            reply = "I'm Chaty G 1! Ask me about science, therapy, controls, or tell me to run a simulation. For detailed theory visuals, try the theory tabs."
        st.session_state.chat_history_g1.append({"role":"chaty", "text": reply})

    # --- Render Chat ---
    for msg in st.session_state.chat_history_g1[-8:]:
        if msg["role"]=="user":
            st.markdown(f"<div style='text-align:right;color:#FFD700;background:#21264b;border-radius:15px 15px 5px 15px;margin:5px;padding:8px;'>{msg['text']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align:left;color:#a3c2fd;background:#192041;border-radius:15px 15px 15px 5px;margin:5px;padding:8px;'>{msg['text']}</div>", unsafe_allow_html=True)
    if st.button("Clear Chaty G 1 Chat"):
        st.session_state.chat_history_g1 = []

    # --- Trigger Theory Simulation from Chaty G 1 ---
    if st.session_state.get("run_theory_from_chatg1", False):
        st.session_state["run_theory_from_chatg1"] = False
        st.success("Running Theory Simulation below!")
        st.markdown("### Step 1: Destroy the Gene")
        fade_steps = np.linspace(0, 1, 10)
        for f in fade_steps:
            fig_dna = plot_dna_helix(fade=f)
            st.plotly_chart(fig_dna, use_container_width=True)
            time.sleep(0.07)
        st.markdown("The gene (DNA helix) fades and breaks (red), representing disabling the faulty component.")

        st.markdown("### Step 2: Boost the Muscles")
        boost_levels = np.linspace(1, 3, 10)
        for b in boost_levels:
            fig_muscle = plot_muscle_boost(boost=b)
            st.plotly_chart(fig_muscle, use_container_width=True)
            time.sleep(0.07)
        st.markdown("Muscle graph increases, representing the stimulation of regeneration.")

        st.markdown("### Step 3: Regenerate")
        progress_steps = np.linspace(0, 1, 20)
        for p in progress_steps:
            fig_regen = plot_regeneration_curve(progress=p)
            st.plotly_chart(fig_regen, use_container_width=True)
            time.sleep(0.07)
        st.markdown("Regeneration curve grows from 0% to 100% (green).")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#FFD700;font-size:1.1em;'>
For farther explanation ask waad naser.<br>
Instagram: <b>@waado__o</b><br>
<i>“A mind that sees too far ahead trapped by its own vision.”</i>
</div>
""", unsafe_allow_html=True)
