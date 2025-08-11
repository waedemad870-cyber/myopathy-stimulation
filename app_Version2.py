import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Collagen 6A3 Myopathy Simulator", page_icon="🧬", layout="wide")

# --- Custom Theme/Style ---
st.markdown("""
<style>
body, .main { background: linear-gradient(135deg, #0d102b 0%, #181b25 50%, #0f1419 100%);}
.stTabs [data-baseweb="tab"] {background:linear-gradient(145deg,#192041,#21264b)!important;color:#FFD700!important;}
.stTabs [aria-selected="true"] {background:linear-gradient(145deg,#21264b,#2a3052)!important;color:#ff5050!important;}
h1, h2, h3, h4, h5, h6 { color: #a3c2fd !important;}
.stButton>button { background: linear-gradient(90deg, #21264b, #FFD700 80%); color: #181b25 !important; font-size:1.1em; font-weight:bold; border-radius:8px; border:2px solid #a3c2fd;}
.stButton>button:hover { box-shadow: 0 0 20px #FFD700;}
.info-card {background:linear-gradient(145deg,#192041,#21264b);color:#FFD700;border-radius:16px;padding:16px;margin:10px 0;border:1.5px solid #a3c2fd;}
.lab-card {background:linear-gradient(90deg,#21264b,#181b25);color:#FFD700;border-radius:10px;padding:10px;margin:10px 0;}
.explain-popup {background:linear-gradient(145deg,#21264b,#0d102b);color:#FFD700;border-radius:14px;padding:18px;margin:10px 0;box-shadow:0 0 10px #FFD700;border:2px solid #FFD700;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🧬 Collagen 6A3 Myopathy Simulator</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Session State ---
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "exp_state" not in st.session_state: st.session_state.exp_state = dict(dna="normal", muscle="normal", protein="normal", compound="normal")
if "theory_step" not in st.session_state: st.session_state.theory_step = 0
if "last_sim" not in st.session_state: st.session_state.last_sim = False

# --- Tabs ---
tab_sim, tab_theory, tab_lab, tab_chem, tab_edu, tab_chat, tab_info = st.tabs([
    "Simulation", "Theory", "Biology Lab", "Chemistry Lab", "Education", "Chaty G 1", "Info"
])

# --- SIM TAB ---
with tab_sim:
    st.markdown("### Simulation Controls")
    try:
        t_rate = st.slider("Transcription Rate (mRNA production)", 0.1, 20.0, 5.0, 0.1)
        d_rate = st.slider("Degradation Rate (mRNA breakdown)", 0.01, 1.0, 0.1, 0.01)
        r_rate = st.slider("Regeneration Rate (muscle recovery)", 0.01, 1.0, 0.05, 0.01)
        sim_time = st.slider("Simulation Time (hours)", 10, 100, 50, 1)
        steps = st.slider("Time Steps (resolution)", 100, 1000, 500, 10)
    except Exception:
        st.warning("Input error: Sliders reset to defaults.")
        t_rate, d_rate, r_rate, sim_time, steps = 5, 0.1, 0.05, 50, 500

    fiber_type = st.selectbox("Muscle Fiber Type", ["Slow-Twitch (Type I)", "Fast-Twitch (Type II)", "Mixed"])
    mut_severity = st.selectbox("Mutation Severity", ["Mild", "Moderate", "Severe"], index=1)

    enable_therapy = st.checkbox("Enable Gene Therapy", value=False)
    if enable_therapy:
        therapy_start = st.slider("Therapy Start Time (hours)", 0, sim_time, int(sim_time//2), 1)
        therapy_duration = st.slider("Therapy Duration (hours)", 1, int(sim_time//2), 10, 1)
        therapy_boost = st.slider("Therapy Boost Factor", 1.0, 5.0, 2.0, 0.1)
    else:
        therapy_start = therapy_duration = therapy_boost = 0

    enable_damage = st.checkbox("Simulate Damage Event", value=False)
    if enable_damage:
        damage_time = st.slider("Damage Event Time (hours)", 0, sim_time, int(sim_time*0.75), 1)
        damage_severity = st.slider("Damage Severity", 0.01, 0.9, 0.3, 0.01)
    else:
        damage_time = damage_severity = 0

    st.markdown("---")
    run_sim = st.button("Run Simulation", key="run_sim_btn")
    if run_sim or st.session_state.last_sim:
        st.session_state.last_sim = True
        dt = sim_time / steps
        t = np.linspace(0, sim_time, steps)
        mRNA = np.zeros(steps); mRNA[0]=1.0
        for i in range(1, steps):
            mRNA[i]=max(mRNA[i-1]+(t_rate-d_rate*mRNA[i-1])*dt,0)
        if enable_therapy and therapy_duration>0:
            start_idx = int(therapy_start/dt)
            end_idx = min(int((therapy_start+therapy_duration)/dt), steps-1)
            mRNA[start_idx:end_idx] *= therapy_boost
        health = np.ones(steps)
        for i in range(1, steps):
            loss = (1.0-mRNA[i])*0.03 if mRNA[i]<1 else 0
            health[i] = min(max(health[i-1]-loss,0)+(r_rate*(1-health[i-1]) if health[i-1]<1 else 0),1)
        if enable_damage and damage_severity>0:
            dmg_idx = int(damage_time/dt)
            health[dmg_idx:] *= (1-damage_severity)
        protein = np.zeros(steps); protein[0]=1.0
        for i in range(1, steps):
            protein[i]=max(protein[i-1]+(mRNA[i]*0.5-protein[i-1]*0.1)*dt,0)
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=mRNA, name="mRNA", line=dict(color="#a3c2fd", width=3)))
        fig.add_trace(go.Scatter(x=t, y=health, name="Health", line=dict(color="#FFD700", width=3)))
        fig.add_trace(go.Scatter(x=t, y=protein, name="Protein", line=dict(color="#00ff88", width=3)))
        fig.update_layout(template="plotly_dark", height=400, title="Simulation Results", plot_bgcolor="#181b25")
        fig.update_xaxes(title="Time (hours)")
        fig.update_yaxes(title="Concentration / Health / Protein")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div class='info-card'>Min Health: {:.3f} | Max Health: {:.3f} | Max Protein: {:.3f}</div>".format(np.min(health),np.max(health),np.max(protein)), unsafe_allow_html=True)
        if st.button("Explain Simulation Graph", key="explain_sim"):
            st.info("This graph shows mRNA, muscle health, and protein over time. Adjust parameters to see effects.")

# --- THEORY TAB ---
with tab_theory:
    st.markdown("### Theory Simulation: Destroy → Boost → Regenerate")
    step_btn = st.button("Next Theory Step")
    if step_btn:
        st.session_state.theory_step = (st.session_state.theory_step+1)%4
    step = st.session_state.theory_step
    n = 100; x=np.linspace(0, 4*np.pi,n); y1=np.sin(x); y2=np.sin(x+np.pi); z=np.linspace(-1,1,n)
    if step==0:
        fig_dna = go.Figure()
        fig_dna.add_trace(go.Scatter3d(x=x, y=y1, z=z, mode='lines', line=dict(color="#a3c2fd", width=7)))
        fig_dna.add_trace(go.Scatter3d(x=x, y=y2, z=z, mode='lines', line=dict(color="#FFD700", width=7)))
        fig_dna.update_layout(template="plotly_dark", scene=dict(bgcolor="#181b25"), title="DNA Normal")
        st.plotly_chart(fig_dna, use_container_width=True)
        st.info("Step 1: DNA is healthy. Click 'Next Theory Step' to destroy the gene.")
    elif step==1:
        fig_dna = go.Figure()
        fig_dna.add_trace(go.Scatter3d(x=x, y=y1, z=z, mode='lines', line=dict(color="#ff5050", width=7)))
        fig_dna.add_trace(go.Scatter3d(x=x, y=y2, z=z, mode='lines', line=dict(color="#FFD700", width=7)))
        fig_dna.update_layout(template="plotly_dark", scene=dict(bgcolor="#181b25"), title="DNA Destroyed")
        st.plotly_chart(fig_dna, use_container_width=True)
        st.warning("Step 2: DNA destroyed. Click 'Next Theory Step' to boost muscles.")
    elif step==2:
        t_muscle = np.linspace(0, 2*np.pi, 200)
        amp = np.linspace(1, 5, 200)
        muscle_wave = amp*np.sin(t_muscle)
        fig_muscle = go.Figure()
        fig_muscle.add_trace(go.Scatter(x=t_muscle, y=muscle_wave, line=dict(color="#FFD700", width=4)))
        fig_muscle.update_layout(template="plotly_dark", plot_bgcolor="#181b25", title="Muscle Boost")
        st.plotly_chart(fig_muscle, use_container_width=True)
        st.success("Step 3: Muscles boosted. Click 'Next Theory Step' to regenerate.")
    elif step==3:
        t_regen = np.linspace(0,1,100)
        regen_curve = 1-np.exp(-6*t_regen)
        fig_regen = go.Figure()
        fig_regen.add_trace(go.Scatter(x=t_regen, y=regen_curve, line=dict(color="#00ff88", width=5)))
        fig_regen.update_layout(template="plotly_dark", plot_bgcolor="#181b25", title="Regeneration")
        st.plotly_chart(fig_regen, use_container_width=True)
        st.success("Step 4: Regeneration complete! Click 'Next Theory Step' to restart.")

# --- BIOLOGY LAB TAB ---
with tab_lab:
    st.markdown("### Interactive Biology Lab")
    exp_type = st.selectbox("Choose an experiment:", ["DNA Helix","Muscle Fiber","Protein Structure"])
    if exp_type=="DNA Helix":
        n = 120; x=np.linspace(0, 4*np.pi,n); y1=np.sin(x); y2=np.sin(x+np.pi); z=np.linspace(-1,1,n)
        fig = go.Figure()
        color = "#a3c2fd" if st.session_state.exp_state["dna"] == "normal" else "#ff5050"
        fig.add_trace(go.Scatter3d(x=x, y=y1, z=z, mode='lines', line=dict(color=color, width=7)))
        fig.add_trace(go.Scatter3d(x=x, y=y2, z=z, mode='lines', line=dict(color="#FFD700", width=7)))
        fig.update_layout(template="plotly_dark", scene=dict(bgcolor="#181b25"), title="DNA Helix")
        st.plotly_chart(fig, use_container_width=True)
        col1, col2 = st.columns(2)
        if col1.button("Mutate DNA"):
            st.session_state.exp_state["dna"]="mutated"
        if col2.button("Repair DNA"):
            st.session_state.exp_state["dna"]="normal"
        if st.session_state.exp_state["dna"]=="mutated":
            st.info("DNA mutated! Red strand indicates damage.")
        else: st.success("DNA is healthy and blue.")
        st.markdown("<div class='lab-card'><b>Guide:</b> DNA mutation can cause muscle disease. Try both buttons.</div>", unsafe_allow_html=True)
    elif exp_type=="Muscle Fiber":
        x_muscle = np.linspace(0, 6*np.pi, 400)
        fatigue = 0.05 if st.session_state.exp_state["muscle"]=="fatigued" else 0.01
        amp = 2 if st.session_state.exp_state["muscle"]=="stimulated" else 1
        y_muscle = amp*np.sin(x_muscle)*np.exp(-fatigue*x_muscle)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_muscle, y=y_muscle, line=dict(color="#FFD700", width=3)))
        fig.update_layout(template="plotly_dark", plot_bgcolor="#181b25", title="Muscle Fiber")
        st.plotly_chart(fig, use_container_width=True)
        col1, col2 = st.columns(2)
        if col1.button("Stimulate Muscle"):
            st.session_state.exp_state["muscle"]="stimulated"
        if col2.button("Fatigue Muscle"):
            st.session_state.exp_state["muscle"]="fatigued"
        if st.session_state.exp_state["muscle"]=="stimulated":
            st.success("Muscle stimulated! Higher amplitude.")
        elif st.session_state.exp_state["muscle"]=="fatigued":
            st.warning("Muscle fatigued! Wave fades.")
        else: st.info("Muscle is in a normal resting state.")
        st.markdown("<div class='lab-card'><b>Guide:</b> Muscle stimulation causes contraction; fatigue causes decline.</div>", unsafe_allow_html=True)
    elif exp_type=="Protein Structure":
        theta = np.linspace(0,2*np.pi,80)
        r = 1+0.2*np.sin(6*theta)
        if st.session_state.exp_state["protein"]=="denatured": r = 1+0.7*np.sin(2*theta)
        x_pro = r*np.cos(theta)
        y_pro = r*np.sin(theta)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_pro, y=y_pro, fill="toself", line=dict(color="#00ff88", width=3)))
        fig.update_layout(template="plotly_dark", plot_bgcolor="#181b25", title="Protein Structure")
        st.plotly_chart(fig, use_container_width=True)
        col1, col2 = st.columns(2)
        if col1.button("Denature Protein"):
            st.session_state.exp_state["protein"]="denatured"
        if col2.button("Synthesize Protein"):
            st.session_state.exp_state["protein"]="normal"
        if st.session_state.exp_state["protein"]=="denatured":
            st.warning("Protein denatured! Shape disrupted.")
        else: st.success("Protein synthesized! Correct structure.")
        st.markdown("<div class='lab-card'><b>Guide:</b> Proteins fold for function. Denature to see loss.</div>", unsafe_allow_html=True)

# --- CHEM LAB TAB ---
with tab_chem:
    st.markdown("### Chemistry Lab")
    exp_type = st.selectbox("Choose a chemistry experiment:", ["Chemistry Compound","Periodic Table"])
    if exp_type=="Chemistry Compound":
        y = [0, 0.6, 0.6]; x=[0, 1, -1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers+lines", marker=dict(size=[24,16,16], color=["#00ff88","#FFD700","#FFD700"]), line=dict(color="#a3c2fd", width=2)))
        fig.update_layout(template="plotly_dark", plot_bgcolor="#181b25", title="Water Molecule (H₂O)")
        st.plotly_chart(fig, use_container_width=True)
        col1, col2 = st.columns(2)
        if col1.button("Ionize Molecule"):
            st.session_state.exp_state["compound"]="ionized"
        if col2.button("Combine Molecules"):
            st.session_state.exp_state["compound"]="normal"
        if st.session_state.exp_state["compound"]=="ionized":
            st.info("Molecule ionized! Charge added.")
        else: st.success("Molecules combined! Back to neutral.")
        st.markdown("<div class='lab-card'><b>Guide:</b> Try both actions to see effects on the molecule.</div>", unsafe_allow_html=True)
    elif exp_type=="Periodic Table":
        st.image("https://upload.wikimedia.org/wikipedia/commons/0/0c/Periodic_table_large.png", use_column_width=True)
        st.markdown("<div class='lab-card'><b>Guide:</b> Use the periodic table to select elements for chemistry experiments.</div>", unsafe_allow_html=True)

# --- EDUCATION TAB ---
with tab_edu:
    st.markdown("### Educational Resources")
    st.markdown("""
    **Collagen VI Chains:**  
    - **COL6A1:** Stability and structure  
    - **COL6A2:** Fiber assembly  
    - **COL6A3:** Most crucial for muscle health (focus for myopathy)  
    **Myopathy:** Muscle disease often caused by gene mutations.  
    **Genes → Protein → Muscle:**  
    - Genes encode mRNA  
    - mRNA → protein synthesis  
    - Proteins (like collagen) build/repair muscle  
    **Mutation Effects:** Mutation in COL6A3 leads to severe muscle weakness, slow regeneration, and poor repair.
    **Gene Therapy:** Boosts mRNA; can help recover muscle health.
    """)

# --- CHAT BOT TAB ---
with tab_chat:
    st.markdown("### 🤖 Chaty G 1 Assistant (Ask Anything!)")
    chat_input = st.text_input("Type your message:", value="", key="chat_input_box")
    send_btn = st.button("Send 🚀", key="send_btn")
    def chaty_g1_respond(msg):
        msg_l = msg.lower().strip()
        if "run theory" in msg_l:
            st.session_state.theory_step = 0
            return "🧬 Theory simulation started! Go to Theory tab."
        if "experiment" in msg_l or "lab" in msg_l:
            return "🧪 Go to Biology/Chemistry Lab tabs for DNA, muscle, protein, chemistry, and periodic table experiments."
        if "hello" in msg_l or "hi" in msg_l or "hey" in msg_l:
            return "👋 Hi! I'm Chaty G 1. Want to try a simulation, experiment, or see theory steps?"
        if "dna" in msg_l:
            return "🧬 DNA is the blueprint for life. Try mutating or repairing it in Biology Lab!"
        if "protein" in msg_l:
            return "🧪 Proteins fold into complex shapes. Try synthesizing or denaturing in the Biology Lab!"
        if "muscle" in msg_l:
            return "💪 Muscles contract and fatigue. Stimulate or fatigue a muscle fiber in the Biology Lab."
        if "chemistry" in msg_l or "compound" in msg_l:
            return "🧪 Chemistry compounds are the building blocks. Try ionizing or combining molecules in Chemistry Lab."
        if "periodic" in msg_l or "table" in msg_l:
            return "🧪 Periodic table is available in Chemistry Lab. Pick elements for experiments!"
        if "quote" in msg_l or "vision" in msg_l:
            return "“A mind that sees too far ahead trapped by its own vision.”"
        if "waad" in msg_l or "waad naser" in msg_l:
            return "🧑‍🔬 For farther explanation ask waad naser. Instagram: @waado__o"
        if "bye" in msg_l or "exit" in msg_l:
            return "👋 Goodbye! See you next session."
        if len(st.session_state.chat_history)>0 and "?" not in msg_l and len(msg_l)<80:
            return f"🤖 You said: '{msg_l}'. Tell me more or ask a question."
        return "🤖 Ask me about simulations, theory, experiments, or guides. For farther explanation ask waad naser. Instagram: @waado__o"
    if send_btn and chat_input.strip():
        st.session_state.chat_history.append({'sender':'user','text':chat_input})
        reply = chaty_g1_respond(chat_input)
        st.session_state.chat_history.append({'sender':'chaty_g1','text':reply})
        st.session_state["chat_input_box"] = ""
    st.markdown("---")
    st.markdown("<div style='max-height:260px;overflow-y:auto;background:#141627;border-radius:10px;padding:10px;'>"+\
        "".join([f"<div style='text-align:{'right' if m['sender']=='user' else 'left'};color:{'#FFD700' if m['sender']=='user' else '#a3c2fd'};'>{m['text']}</div>" for m in st.session_state.chat_history])+"</div>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat", key="clear_chat_btn"):
        st.session_state.chat_history = []

# --- INFO TAB ---
with tab_info:
    st.markdown("## About This App")
    st.markdown("""
    - Multiscale simulation of Collagen 6A3 gene, muscle, protein, and chemistry.
    - Theory simulation: stepwise DNA destruction, muscle boost, and regeneration.
    - Interactive biology and chemistry labs: mutate DNA, fatigue muscle, denature proteins, combine/ionize molecules.
    - Chaty G 1 bot: guides, answers, triggers features.
    - All graphics: Plotly, dark theme, rich colors.
    - Attribution: For farther explanation ask waad naser. Instagram: **@waado__o**
    """)
    st.markdown("""<div class='info-card'><b>“A mind that sees too far ahead trapped by its own vision.”</b></div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#FFD700;font-size:1.1em;'>
For farther explanation ask waad naser.<br>
Instagram: <b>@waado__o</b><br>
<i>“A mind that sees too far ahead trapped by its own vision.”</i>
</div>
""", unsafe_allow_html=True)
</div>
""", unsafe_allow_html=True)

