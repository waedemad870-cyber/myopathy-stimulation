import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Setup & Custom Theme ---
st.set_page_config(
    page_title="Collagen 6A3 Myopathy Simulator",
    page_icon="🧬",
    layout="wide"
)

# Custom dark theme and accent styles
st.markdown("""
<style>
body, .main {
    background: linear-gradient(135deg, #0d102b 0%, #181b25 50%, #0f1419 100%);
}
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
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown(
    "<h1 style='text-align:center;'>🧬 Collagen 6A3 Myopathy Simulator</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align:center;'>Gene Expression · Muscle Health · Regeneration Modeling</h3>",
    unsafe_allow_html=True
)
st.markdown("---")

# --- Feature Buttons Section ---
colA, colB, colC = st.columns([1, 1, 2])
with colA:
    if st.button("💡 What is Myopathy & Collagen VI?"):
        st.markdown("""
        <div class="explain-popup">
        <b>What is Myopathy?</b><br>
        Myopathy is a disease affecting muscle fibers and strength, often caused by genetic mutations.<br><br>
        <b>Collagen VI Chains:</b><br>
        <ul>
        <li><b>A1 (COL6A1):</b> Stability and structure</li>
        <li><b>A2 (COL6A2):</b> Helps assemble collagen fibers</li>
        <li><b>A3 (COL6A3):</b> <span style='color:#FFD700;font-weight:bold;'>MOST crucial for muscle health!</span><br>
        Mutations here cause severe disorders (Ullrich CMD, Bethlem myopathy).
        <br><br>
        <b>COL6A3 dysfunction:</b> Leads to poor muscle regeneration, mitochondrial failure, and increased cell death.
        </li>
        </ul>
        <b>Why focus on A3?</b> It’s most often mutated in congenital muscle diseases and a key target for research and therapy!
        </div>
        """, unsafe_allow_html=True)
with colB:
    if st.button("🛠 About This App"):
        st.markdown("""
        <div class="explain-popup">
        <b>This Simulator helps you:</b>
        <ul>
        <li>Model gene expression for Collagen 6A3</li>
        <li>Visualize muscle health and regeneration</li>
        <li>Test effects of severity, fiber type, therapy, and injury</li>
        <li>Export results, view interactive charts, and chat with the built-in assistant!</li>
        </ul>
        <b>How to use?</b> Adjust sidebar controls, run a simulation, and explore the tabs!
        </div>
        """, unsafe_allow_html=True)

with colC:
    st.markdown("""
    <div class='info-card'>
    <b>Features:</b> Simulation · Therapy · Injury · Multi-fiber types · Visual Dashboard · AI Assistant
    </div>
    """, unsafe_allow_html=True)

# --- Sidebar: User Inputs ---
st.sidebar.markdown("<h2 style='color:#FFD700;'>Simulation Controls</h2>", unsafe_allow_html=True)
transcription_rate = st.sidebar.slider("Transcription Rate (mRNA production)", 0.1, 20.0, 5.0, 0.1)
degradation_rate = st.sidebar.slider("Degradation Rate (mRNA breakdown)", 0.01, 1.0, 0.1, 0.01)
regeneration_rate = st.sidebar.slider("Regeneration Rate (muscle recovery)", 0.01, 1.0, 0.05, 0.01)
simulation_time = st.sidebar.slider("Simulation Time (hours)", 10, 100, 50, 1)
time_steps = st.sidebar.slider("Time Steps (resolution)", 100, 1000, 500, 10)

st.sidebar.markdown("—")
fiber_type = st.sidebar.selectbox(
    "Muscle Fiber Type",
    ["Slow-Twitch (Type I)", "Fast-Twitch (Type II)", "Mixed"]
)
mutation_severity = st.sidebar.selectbox(
    "Mutation Severity",
    ["Mild", "Moderate", "Severe"], index=1
)

st.sidebar.markdown("—")
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
- Install dependencies: pip install streamlit numpy pandas plotly  
- Run: streamlit run app.py
- Try Streamlit Cloud for easy deployment!
""")

# --- Simulation Functions ---
def simulate_gene_expression(transcription_rate, degradation_rate, simulation_time, time_steps):
    dt = simulation_time / time_steps
    t = np.linspace(0, simulation_time, time_steps)
    mRNA = np.zeros(time_steps)
    mRNA[0] = 1.0
    for i in range(1, time_steps):
        dmRNA = transcription_rate - degradation_rate * mRNA[i-1]
        mRNA[i] = max(mRNA[i-1] + dmRNA * dt, 0)
    return mRNA, t

def simulate_gene_therapy(mRNA, t, start_time, duration, boost_factor):
    mRNA_therapy = mRNA.copy()
    start_idx = np.searchsorted(t, start_time)
    end_idx = np.searchsorted(t, start_time + duration)
    for i in range(start_idx, min(end_idx, len(mRNA))):
        mRNA_therapy[i] *= boost_factor
    return mRNA_therapy

def simulate_muscle_health(mRNA, regeneration_rate, time_steps):
    health = np.ones(time_steps)
    regen = np.zeros(time_steps)
    for i in range(1, time_steps):
        if mRNA[i] < 1.0:
            loss = (1.0 - mRNA[i]) * 0.03
            health[i] = max(health[i-1] - loss, 0.0)
        else:
            health[i] = min(health[i-1] + 0.01, 1.0)
        if health[i] < 1.0:
            regen[i] = regeneration_rate * (1.0 - health[i])
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
    return {
        "min_health": min_health,
        "max_health": max_health,
        "recovery_time": recovery_time
    }

def export_csv(data_dict):
    df = pd.DataFrame(data_dict)
    return df.to_csv(index=False)

# --- Tabs ---
tab_sim, tab_explain, tab_guide, tab_chat = st.tabs([
    "Simulation Results", "Scientific Concepts", "User Guide", "Chaty G 1 Assistant"
])

# --- Simulation Tab ---
with tab_sim:
    st.markdown("## 🧪 Simulation Visualization")
    run_sim = st.button("🚀 Run Simulation", key="run_sim_btn")
    if run_sim or st.session_state.get("last_sim", False):
        st.session_state["last_sim"] = True
        severity = {"Mild": (0.9, 1.1), "Moderate": (0.7, 1.3), "Severe": (0.5, 1.6)}
        tr_mult, dr_mult = severity[mutation_severity]
        adj_tr = transcription_rate * tr_mult
        adj_dr = degradation_rate * dr_mult

        # Fiber type effect
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

        # --- Plotly Visualization ---
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "mRNA Concentration", "Muscle Health",
                "Regeneration Effect", "Collagen Protein"
            ),
            specs=[[{}, {}], [{}, {}]],
            vertical_spacing=0.12, horizontal_spacing=0.09
        )
        fig.add_trace(go.Scatter(x=t, y=mRNA, line=dict(color="#a3c2fd", width=2), name="mRNA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=health, line=dict(color="#FFD700", width=2), name="Health"), row=1, col=2)
        fig.add_trace(go.Scatter(x=t, y=regen, line=dict(color="#ff5050", width=2, dash="dot"), name="Regeneration"), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=protein, line=dict(color="#00ff88", width=2), name="Protein"), row=2, col=2)
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#181b25",
            paper_bgcolor="#181b25",
            font_color="#a3c2fd",
            height=600
        )
        for i in range(1,3):
            fig.update_xaxes(title_text="Time (hours)", row=i, col=1)
            fig.update_xaxes(title_text="Time (hours)", row=i, col=2)
        fig.update_yaxes(title_text="mRNA", row=1, col=1)
        fig.update_yaxes(title_text="Health (0-1)", row=1, col=2)
        fig.update_yaxes(title_text="Regeneration", row=2, col=1)
        fig.update_yaxes(title_text="Protein", row=2, col=2)
        st.plotly_chart(fig, use_container_width=True)

        # --- Summary Stats ---
        st.markdown(
            f"<div class='info-card'>"
            f"<b>Min Health:</b> {stats['min_health']:.3f} &nbsp;&nbsp;"
            f"<b>Max Health:</b> {stats['max_health']:.3f} &nbsp;&nbsp;"
            f"<b>Time to Full Regeneration:</b> {stats['recovery_time']:.1f} hours"
            f"</div>", unsafe_allow_html=True
        )

        # --- Download ---
        data_dict = {
            "Time": t,
            "mRNA": mRNA,
            "Health": health,
            "Regeneration": regen,
            "Protein": protein
        }
        csv_data = export_csv(data_dict)
        st.download_button("Download Simulation CSV", csv_data, "simulation_results.csv", "text/csv")

# --- Explanation Tab ---
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
    
    """)
    st.markdown("""
    *References:*  
    - Bönnemann, C.G. "The collagen VI-related myopathies." Handbook of Clinical Neurology (2011)  
    - Lampe, A.K., & Bushby, K.M. "Collagen VI related muscle disorders." Journal of Medical Genetics (2005)
    """)

# --- User Guide Tab ---
with tab_guide:
    st.markdown("## 📖 User Guide")
    st.markdown("""
    *How to Use This App:*
    1. Adjust parameters in the sidebar (transcription, degradation, regeneration, etc.)
    2. Click through tabs to explore simulation results, scientific background, and chatbot.
    3. Download your results as CSV for further analysis.

    *Tips:*
    - Enable gene therapy or damage events for advanced modeling.
    - Use the chatbot for interactive help and navigation.

    *Deployment:*
    - Run locally: streamlit run app.py
    - Deploy to [Streamlit Cloud](https://streamlit.io/cloud)
    """)

# --- Chaty G 1 Tab ---
with tab_chat:
    st.markdown("## 🤖 Chaty G 1 - AI Assistant")
    # --- Chat UI State ---
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    chat_container = st.container()
    with chat_container:
        st.markdown('<div style="max-height:300px;overflow-y:auto;background:#141627;border-radius:10px;padding:10px;">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg['sender'] == 'user':
                st.markdown(f'<div style="text-align:right;color:#FFD700;background:#21264b;border-radius:15px 15px 5px 15px;margin:5px;">{msg["text"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="text-align:left;color:#a3c2fd;background:#192041;border-radius:15px 15px 15px 5px;margin:5px;">{msg["text"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    # --- Chat Input ---
    col1, col2 = st.columns([4,1])
    with col1:
        chat_input = st.text_input("Type your message...", key="chat_input_box")
    with col2:
        send_btn = st.button("Send 🚀")
    # --- Chatbot Logic ---
    def chaty_g1_respond(msg):
        msg_l = msg.lower().strip()
        if any(k in msg_l for k in ['dashboard','home','main']):
            return ("🏠 Switching to the Simulation tab!", "navigate_sim")
        if any(k in msg_l for k in ['simulation','gene','run']):
            return ("🧪 Opening Simulation Visualization...", "navigate_sim")
        if 'reset' in msg_l:
            return ("🔄 Resetting simulation data!", "reset_sim")
        if 'exit' in msg_l:
            return ("👋 Goodbye! Thanks for using Collagen 6A3 Simulator.", "exit")
        if any(k in msg_l for k in ['collagen','6a3']):
            return ("🧬 Collagen 6A3 encodes a major muscle protein. Mutations cause myopathy. Try running a simulation!", None)
        if 'help' in msg_l or 'guide' in msg_l:
            return ("📖 Adjust sidebar parameters, run simulation, and explore results. Ask me anything!", None)
        if 'therapy' in msg_l or 'treatment' in msg_l:
            return ("🔬 Gene therapy boosts mRNA, simulating treatment response. Enable it in the sidebar.", None)
        return ("🤖 I can help with navigation, science Q&A, or usage tips. Try: 'run simulation', 'reset', 'what is collagen 6A3?'", None)
    if send_btn and chat_input:
        st.session_state.chat_history.append({'sender':'user','text':chat_input})
        reply, action = chaty_g1_respond(chat_input)
        st.session_state.chat_history.append({'sender':'chaty_g1','text':reply})
        if action == "navigate_sim":
            st.experimental_set_query_params(tab="Simulation Results")
        if action == "reset_sim":
            st.session_state.chat_history = []
        if action == "exit":
            st.write("👋 Goodbye!")
        st.experimental_rerun()
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()
