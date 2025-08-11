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
.graph-explain-btn {
    background: linear-gradient(90deg, #1e2a74, #FFD700);
    color: #21264b;
    border-radius: 8px;
    border: none;
    padding: 8px 16px;
    font-weight: bold;
    margin-top: 8px;
    margin-bottom: 8px;
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
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align:center;'>🧬 Collagen 6A3 Myopathy Simulator</h1><h3 style='text-align:center;'>Gene Expression · Muscle Health · Regeneration Modeling</h3>",
    unsafe_allow_html=True
)
st.markdown("---")

# --- Utility: Error-Resistant Parameter Validation ---
def safe_slider(*args, **kwargs):
    try:
        return st.sidebar.slider(*args, **kwargs)
    except Exception as e:
        st.sidebar.error(f"Slider error: {e}")
        return kwargs.get('value', 0)

def safe_selectbox(*args, **kwargs):
    try:
        return st.sidebar.selectbox(*args, **kwargs)
    except Exception as e:
        st.sidebar.error(f"Select error: {e}")
        opts = kwargs.get('options', [])
        return opts[0] if opts else None

def safe_checkbox(*args, **kwargs):
    try:
        return st.sidebar.checkbox(*args, **kwargs)
    except Exception as e:
        st.sidebar.error(f"Checkbox error: {e}")
        return False

# --- Sidebar Controls: Safe Inputs ---
st.sidebar.markdown("<h2 style='color:#FFD700;'>Simulation Controls</h2>", unsafe_allow_html=True)
transcription_rate = safe_slider("Transcription Rate (mRNA production)", 0.1, 20.0, 5.0, 0.1)
degradation_rate = safe_slider("Degradation Rate (mRNA breakdown)", 0.01, 1.0, 0.1, 0.01)
regeneration_rate = safe_slider("Regeneration Rate (muscle recovery)", 0.01, 1.0, 0.05, 0.01)
simulation_time = safe_slider("Simulation Time (hours)", 10, 100, 50, 1)
time_steps = safe_slider("Time Steps (resolution)", 100, 1000, 500, 10)

st.sidebar.markdown("—")
fiber_type = safe_selectbox(
    "Muscle Fiber Type",
    ["Slow-Twitch (Type I)", "Fast-Twitch (Type II)", "Mixed"]
)
mutation_severity = safe_selectbox(
    "Mutation Severity",
    ["Mild", "Moderate", "Severe"], index=1
)

st.sidebar.markdown("—")
enable_gene_therapy = safe_checkbox("Enable Gene Therapy", value=False)
if enable_gene_therapy:
    therapy_start_time = safe_slider("Therapy Start Time (hours)", 0, simulation_time, int(simulation_time//2), 1)
    therapy_duration = safe_slider("Therapy Duration (hours)", 1, int(simulation_time//2), 10, 1)
    therapy_boost = safe_slider("Therapy Boost Factor", 1.0, 5.0, 2.0, 0.1)
else:
    therapy_start_time = therapy_duration = therapy_boost = 0

enable_damage = safe_checkbox("Simulate Damage Event", value=False)
if enable_damage:
    damage_time = safe_slider("Damage Event Time (hours)", 0, simulation_time, int(simulation_time*0.75), 1)
    damage_severity = safe_slider("Damage Severity", 0.01, 0.9, 0.3, 0.01)
else:
    damage_time = damage_severity = 0

st.sidebar.markdown("---")
st.sidebar.markdown("""
*How to run:*  
- Install dependencies: pip install streamlit numpy pandas plotly  
- Run: streamlit run app.py
- Try Streamlit Cloud for easy deployment!
""")

# --- Main Tabs ---
tab_sim, tab_theory, tab_experiments, tab_explain, tab_guide, tab_chat = st.tabs([
    "Simulation Results", "Theory Simulation", "Biology Lab", "Scientific Concepts", "User Guide", "Chaty G 1 Assistant"
])

# --- Theory Simulation Tab ---
with tab_theory:
    st.markdown("<h2 style='color:#FFD700'>🧬 Theory Simulation Lab</h2>", unsafe_allow_html=True)
    st.markdown("Step through the experimental theory: Destroy the gene → Boost muscles → Regenerate tissue.")
    run_theory = st.button("Run Theory Simulation", key="run_theory_sim_btn")
    if run_theory or st.session_state.get("run_theory", False):
        st.session_state["run_theory"] = True

        col1, col2, col3 = st.columns(3)

        # Step 1: DNA Helix - Destroy animation
        with col1:
            n = 100
            x = np.linspace(0, 4 * np.pi, n)
            y1 = np.sin(x)
            y2 = np.sin(x + np.pi)
            z = np.linspace(-1, 1, n)
            fig_dna = go.Figure()
            fig_dna.add_trace(go.Scatter3d(
                x=x, y=y1, z=z, mode='lines',
                line=dict(color="#a3c2fd", width=6),
                name="Strand 1"
            ))
            fig_dna.add_trace(go.Scatter3d(
                x=x, y=y2, z=z, mode='lines',
                line=dict(color="#FFD700", width=6),
                name="Strand 2"
            ))
            fig_dna.add_trace(go.Scatter3d(
                x=x[n//2:], y=y1[n//2:], z=z[n//2:], mode='lines',
                line=dict(color="#ff5050", width=10),
                name="Break"
            ))
            fig_dna.update_layout(
                title="Step 1: Destroy the Gene",
                template="plotly_dark",
                margin=dict(l=0, r=0, b=0, t=30),
                scene=dict(bgcolor="#181b25", xaxis_title="", yaxis_title="", zaxis_title="")
            )
            st.plotly_chart(fig_dna, use_container_width=True)
            st.markdown("**Visualizing gene destruction:** The DNA helix breaks and fades to red, representing mutation or loss of function.")

        # Step 2: Boost Muscles
        with col2:
            t_muscle = np.linspace(0, 2*np.pi, 200)
            amplitude = np.linspace(1, 3, 200)
            muscle_wave = amplitude * np.sin(t_muscle)
            fig_muscle = go.Figure()
            fig_muscle.add_trace(go.Scatter(
                x=t_muscle, y=muscle_wave,
                line=dict(color="#FFD700", width=3),
                name="Muscle Activity"
            ))
            fig_muscle.update_layout(
                title="Step 2: Boost the Muscles",
                template="plotly_dark",
                plot_bgcolor="#181b25",
                paper_bgcolor="#181b25",
                font_color="#FFD700",
                margin=dict(l=0, r=0, b=0, t=30)
            )
            st.plotly_chart(fig_muscle, use_container_width=True)
            st.markdown("**Muscle boost:** Activity spikes as amplitude increases, representing muscle stimulation or intervention.")

        # Step 3: Regenerate
        with col3:
            t_regen = np.linspace(0, 1, 100)
            regen_curve = 1 - np.exp(-6*t_regen)
            fig_regen = go.Figure()
            fig_regen.add_trace(go.Scatter(
                x=t_regen, y=regen_curve,
                line=dict(color="#00ff88", width=5),
                name="Regeneration"
            ))
            fig_regen.update_layout(
                title="Step 3: Regenerate",
                template="plotly_dark",
                plot_bgcolor="#181b25",
                paper_bgcolor="#181b25",
                font_color="#00ff88",
                margin=dict(l=0, r=0, b=0, t=30),
                xaxis=dict(title="Progress (0-1)"),
                yaxis=dict(title="Regeneration (%)", range=[0,1])
            )
            st.plotly_chart(fig_regen, use_container_width=True)
            st.markdown("**Regeneration:** Green curve shows recovery from 0% to 100%, modeling healing after damage.")

        st.markdown("<div class='info-card'>Theory simulation complete. Want to try experiments? Go to the Biology Lab tab!</div>", unsafe_allow_html=True)

# --- Experiments Tab: Interactive Biology Lab ---
with tab_experiments:
    st.markdown("<h2 style='color:#FFD700'>🧪 Biology Lab: Visual Experiments</h2>", unsafe_allow_html=True)
    st.markdown("Try experiments on genes, muscles, proteins, and compounds. Use the periodic table. Follow the guides below.")
    exp_graphics = st.selectbox("Choose a graphic to experiment with:", [
        "DNA Helix", "Muscle Fiber", "Protein Molecule", "Chemistry Compound", "Periodic Table"
    ])
    if exp_graphics == "DNA Helix":
        st.markdown("### 🧬 DNA Double Helix")
        n = 120
        x = np.linspace(0, 4*np.pi, n)
        y1 = np.sin(x)
        y2 = np.sin(x + np.pi)
        z = np.linspace(-1, 1, n)
        fig_dna_exp = go.Figure()
        fig_dna_exp.add_trace(go.Scatter3d(
            x=x, y=y1, z=z, mode='lines',
            line=dict(color="#a3c2fd", width=6),
            name="Strand 1"
        ))
        fig_dna_exp.add_trace(go.Scatter3d(
            x=x, y=y2, z=z, mode='lines',
            line=dict(color="#FFD700", width=6),
            name="Strand 2"
        ))
        fig_dna_exp.update_layout(
            title="DNA Helix Visualization",
            template="plotly_dark",
            margin=dict(l=0, r=0, b=0, t=30),
            scene=dict(bgcolor="#181b25", xaxis_title="", yaxis_title="", zaxis_title="")
        )
        st.plotly_chart(fig_dna_exp, use_container_width=True)
        st.markdown("**Experiment:** Click below to mutate or repair the DNA strand.")
        col_mutate, col_repair = st.columns(2)
        if col_mutate.button("Mutate DNA"):
            st.info("DNA mutated! Red color indicates genetic damage.")
        if col_repair.button("Repair DNA"):
            st.success("DNA repaired! Back to healthy blue/yellow strands.")
        st.markdown("**Guide:** DNA mutations can cause myopathies. Try mutating and repairing the helix to visualize effects.")

    elif exp_graphics == "Muscle Fiber":
        st.markdown("### 🦾 Muscle Fiber Structure")
        x_muscle = np.linspace(0, 6*np.pi, 300)
        y_muscle = np.sin(x_muscle) * np.exp(-0.1*x_muscle)
        fig_muscle_exp = go.Figure()
        fig_muscle_exp.add_trace(go.Scatter(
            x=x_muscle, y=y_muscle, line=dict(color="#FFD700", width=3),
            name="Muscle Fiber"
        ))
        fig_muscle_exp.update_layout(
            title="Muscle Fiber Visualization",
            template="plotly_dark",
            plot_bgcolor="#181b25",
            paper_bgcolor="#181b25",
            font_color="#FFD700"
        )
        st.plotly_chart(fig_muscle_exp, use_container_width=True)
        st.markdown("**Experiment:** Click below to stimulate or fatigue the muscle.")
        col_stim, col_fatigue = st.columns(2)
        if col_stim.button("Stimulate Muscle"):
            st.info("Muscle stimulated! Amplitude spikes, representing contraction.")
        if col_fatigue.button("Fatigue Muscle"):
            st.warning("Muscle fatigued. Amplitude drops, representing tired tissue.")
        st.markdown("**Guide:** Muscles contract and fatigue with time. Try both to see the fiber dynamics.")

    elif exp_graphics == "Protein Molecule":
        st.markdown("### 🧪 Protein Visualization")
        theta = np.linspace(0, 2*np.pi, 50)
        r = 1 + 0.2*np.sin(6*theta)
        x_pro = r * np.cos(theta)
        y_pro = r * np.sin(theta)
        fig_protein_exp = go.Figure()
        fig_protein_exp.add_trace(go.Scatter(
            x=x_pro, y=y_pro, fill="toself", line=dict(color="#00ff88", width=3),
            name="Protein"
        ))
        fig_protein_exp.update_layout(
            title="Collagen Protein Molecule",
            template="plotly_dark",
            plot_bgcolor="#181b25",
            paper_bgcolor="#181b25",
            font_color="#00ff88"
        )
        st.plotly_chart(fig_protein_exp, use_container_width=True)
        st.markdown("**Experiment:** Click below to denature or synthesize protein.")
        col_denature, col_synth = st.columns(2)
        if col_denature.button("Denature Protein"):
            st.warning("Protein denatured! Shape disrupted, loss of function.")
        if col_synth.button("Synthesize Protein"):
            st.success("Protein synthesized! Function restored.")
        st.markdown("**Guide:** Proteins fold into shapes. Myopathy disrupts structure. Experiment with denaturation and synthesis.")

    elif exp_graphics == "Chemistry Compound":
        st.markdown("### 🧪 Chemistry Compound Visualization")
        fig_water = go.Figure()
        fig_water.add_trace(go.Scatter(
            x=[0, 1, -1], y=[0, 0.6, 0.6], mode="markers+lines",
            marker=dict(size=[24,16,16], color=["#00ff88","#FFD700","#FFD700"]),
            line=dict(color="#a3c2fd", width=2),
            name="H2O"
        ))
        fig_water.update_layout(
            title="Water Molecule (H₂O)",
            template="plotly_dark",
            plot_bgcolor="#181b25",
            paper_bgcolor="#181b25",
            font_color="#00ff88"
        )
        st.plotly_chart(fig_water, use_container_width=True)
        st.markdown("**Experiment:** Click below to ionize or combine molecules.")
        col_ionize, col_combine = st.columns(2)
        if col_ionize.button("Ionize Molecule"):
            st.info("Molecule ionized! Changed to charged state.")
        if col_combine.button("Combine Molecules"):
            st.success("Molecules combined! Try building more complex compounds.")
        st.markdown("**Guide:** Chemistry compounds are building blocks. Try ionizing or combining molecules.")

    elif exp_graphics == "Periodic Table":
        st.markdown("### 🧪 Periodic Table")
        st.image("https://upload.wikimedia.org/wikipedia/commons/0/0c/Periodic_table_large.png", use_column_width=True)
        st.markdown("**Guide:** Use the periodic table to select elements for your biology and chemistry experiments.")

# --- Chaty G 1 Tab ---
with tab_chat:
    st.markdown("## 🤖 Chaty G 1 - AI Assistant")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    chat_container = st.container()
    with chat_container:
        st.markdown('<div style="max-height:400px;overflow-y:auto;background:#141627;border-radius:10px;padding:10px;">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg['sender'] == 'user':
                st.markdown(f'<div style="text-align:right;color:#FFD700;background:#21264b;border-radius:15px 15px 5px 15px;margin:5px;">{msg["text"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="text-align:left;color:#a3c2fd;background:#192041;border-radius:15px 15px 15px 5px;margin:5px;">{msg["text"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([4,1])
    with col1:
        chat_input = st.text_input("Type your message...", key="chat_input_box", value=st.session_state.get("chat_input", ""))
    with col2:
        send_btn = st.button("Send 🚀")
    def chaty_g1_respond(msg):
        msg_l = msg.lower().strip()
        if "run theory" in msg_l:
            st.session_state["run_theory"] = True
            return "🧬 Running your theory simulation now. Check the Theory Simulation tab!"
        if any([k in msg_l for k in ["dna", "protein", "molecule", "periodic", "chemistry"]]):
            return "🧪 Go to the Biology Lab tab to explore DNA, proteins, chemistry, and the periodic table. You can experiment interactively!"
        if "hello" in msg_l or "hi" in msg_l or "hey" in msg_l:
            return "👋 Hi! I'm Chaty G 1. How can I help you explore the Collagen 6A3 simulator today?"
        if "collagen" in msg_l or "6a3" in msg_l:
            return "🧬 Collagen 6A3 is the alpha-3 chain of the collagen VI protein, and is crucial for muscle integrity. Mutations here lead to muscle weakness and myopathy. Ask me about gene therapy, simulation, or muscle health!"
        if "what is myopathy" in msg_l or "myopathy" in msg_l:
            return "🦾 Myopathy is a disease of muscle tissue, often genetic. Collagen VI-related myopathies affect strength and regeneration. You can simulate these effects here!"
        if "health" in msg_l or "muscle" in msg_l:
            return "💪 Muscle health is tracked from 0 (damaged) to 1 (healthy). You can see how it changes based on gene expression, therapy, and injury."
        if "therapy" in msg_l or "treatment" in msg_l:
            return "🧬 Gene therapy boosts mRNA, simulating possible treatments. Enable it in the sidebar and rerun your simulation!"
        if "regeneration" in msg_l:
            return "🔄 Regeneration models the natural healing of muscle after injury. You can change how fast this happens using the sidebar."
        if "download" in msg_l:
            return "📱 You can download the app for iPhone, Android, Mac, or Windows using the buttons above!"
        if "waad" in msg_l or "waad naser" in msg_l:
            return "🧑‍🔬 For farther explanation ask waad naser. Instagram: @waado__o"
        if "quote" in msg_l or "vision" in msg_l:
            return "“A mind that sees too far ahead trapped by its own vision.”"
        if "long" in msg_l or "conversation" in msg_l or "chat" in msg_l:
            return "🤖 I'm designed for longer conversations! Ask me anything, and I'll keep the discussion going!"
        if "bye" in msg_l or "exit" in msg_l:
            return "👋 Goodbye! Feel free to come back to the Collagen 6A3 Myopathy Simulator anytime."
        if len(st.session_state.chat_history) > 0 and "?" not in msg_l and len(msg_l) < 80:
            return f"🤖 You said: '{msg_l}'. Can you tell me more or ask a follow-up question?"
        return "🤖 I'm Chaty G 1! Ask me about collagen, muscle health, gene therapy, or app features. For farther explanation ask waad naser. Instagram: @waado__o"
    if send_btn and chat_input:
        st.session_state.chat_history.append({'sender':'user','text':chat_input})
        reply = chaty_g1_respond(chat_input)
        st.session_state.chat_history.append({'sender':'chaty_g1','text':reply})
        st.session_state["chat_input"] = ""
        st.experimental_rerun()
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#FFD700;font-size:1.1em;'>
For farther explanation ask waad naser.<br>
Instagram: <b>@waado__o</b><br>
<i>“A mind that sees too far ahead trapped by its own vision.”</i>
</div>
""", unsafe_allow_html=True)

