import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from PIL import Image
import yaml
import os
import time
import random
import itertools
from dqn import DQN
from experience_replay import ReplayMemory

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Flappy Bird Deep Q-Learning AI",
    page_icon="🐤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0d1322 100%);
    }
    
    /* Header Gradient */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphic Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
    }
    
    /* Metric styling */
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Action badge */
    .action-badge-flap {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .action-badge-none {
        background: linear-gradient(135deg, #475569 0%, #64748b 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# --- UTILS & MODEL INITIALIZATION ---
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
DEFAULT_MODEL_PATH = os.path.join(RUNS_DIR, "flappybird.pt")

@st.cache_data
def load_default_params():
    if os.path.exists("parameters.yaml"):
        with open("parameters.yaml", "r") as f:
            return yaml.safe_load(f).get("flappybird", {})
    return {
        "alpha": 0.001,
        "gamma": 0.99,
        "epsilon_init": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.9995,
        "replay_memory_size": 100000,
        "mini_batch_size": 32,
        "network_sync_rate": 10,
        "reward_threshold": 1000
    }

def get_policy_network(state_dim=180, action_dim=2, model_path=None):
    """
    Instantiates DQN and dynamically inspects checkpoint weight shapes to avoid dimension mismatch.
    """
    if model_path and os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, weights_only=True, map_location=DEVICE)
            # Inspect first layer weight shape to determine exact input state dim
            first_layer_key = "model.0.weight" if "model.0.weight" in state_dict else "0.weight"
            if first_layer_key in state_dict:
                state_dim = state_dict[first_layer_key].shape[1]
                
            model = DQN(state_dim, action_dim).to(DEVICE)
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception as e:
            st.warning(f"Note loading checkpoint ({e}). Initializing default model.")
            
    return DQN(state_dim, action_dim).to(DEVICE)

# --- HEADER ---
st.markdown('<div class="main-title">🐤 Flappy Bird Deep Q-Learning AI Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Autonomous Reinforcement Learning Agent powered by PyTorch & OpenAI Gymnasium</div>', unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://raw.githubusercontent.com/tensorspace-team/tensorspace/master/assets/img/flappybird.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Mode",
    ["🤖 Live AI Showcase", "🧪 Hyperparameter Training Lab", "🧠 Neural Network Inspector", "🎮 Human Play Guide"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🖥️ Device Info")
st.sidebar.info(f"**PyTorch Device:** `{DEVICE.upper()}`")

# ---------------------------------------------------------
# PAGE 1: LIVE AI SHOWCASE
# ---------------------------------------------------------
if page == "🤖 Live AI Showcase":
    st.markdown("### 🤖 Live AI Gameplay & Real-Time Q-Value Telemetry")
    
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📺 Live Canvas")
        frame_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            start_ai = st.button("▶️ Start AI Agent", use_container_width=True, type="primary")
        with c2:
            stop_ai = st.button("⏹️ Stop", use_container_width=True)
        with c3:
            delay = st.slider("Frame Speed (s)", 0.0, 0.2, 0.03, 0.01)

    with col_right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Live Decision Telemetry")
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            score_metric = st.empty()
        with m_col2:
            action_metric = st.empty()
            
        st.markdown("##### Q-Value Action Probabilities")
        q_chart_placeholder = st.empty()
        
        st.markdown("##### State Vector Features (LIDAR / Distance)")
        state_df_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    if start_ai:
        env = gym.make("FlappyBird-v0", render_mode="rgb_array")
        obs_dim = env.observation_space.shape[0]
        model = get_policy_network(state_dim=obs_dim, model_path=DEFAULT_MODEL_PATH)
        
        state, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0
        
        while not (terminated or truncated):
            # Render frame
            img_arr = env.render()
            frame_placeholder.image(img_arr, use_container_width=True, caption=f"Frame Step: {step_count}")
            
            # Predict action
            state_tensor = torch.tensor(state, dtype=torch.float, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                q_vals = model(state_tensor).squeeze().cpu().numpy()
                action = int(np.argmax(q_vals))
            
            # Update metrics
            total_reward += 1
            step_count += 1
            
            score_metric.metric("Current Score / Reward", f"{int(total_reward)}")
            if action == 1:
                action_metric.markdown('**Current Action:** <span class="action-badge-flap">FLAP 🚀</span>', unsafe_allow_html=True)
            else:
                action_metric.markdown('**Current Action:** <span class="action-badge-none">DO NOTHING 😴</span>', unsafe_allow_html=True)
            
            # Q-Value Bar Chart
            fig_q = go.Figure(go.Bar(
                x=["No Flap (0)", "Flap (1)"],
                y=q_vals,
                marker_color=["#64748b" if action != 0 else "#38bdf8", "#64748b" if action != 1 else "#10b981"],
                text=[f"{v:.3f}" for v in q_vals],
                textposition='auto'
            ))
            fig_q.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                height=180,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#cbd5e1")
            )
            q_chart_placeholder.plotly_chart(fig_q, use_container_width=True)
            
            # State vector display
            df_state = pd.DataFrame({
                "Feature Index": [f"State[{i}]" for i in range(min(10, len(state)))],
                "Value": np.round(state[:10], 4)
            })
            state_df_placeholder.dataframe(df_state, height=180, use_container_width=True)
            
            # Step environment
            state, reward, terminated, truncated, info = env.step(action)
            time.sleep(delay)
            
        env.close()
        st.success(f"🎮 Episode Ended! Final Reward Score: {int(total_reward)}")

# ---------------------------------------------------------
# PAGE 2: HYPERPARAMETER TRAINING LAB
# ---------------------------------------------------------
elif page == "🧪 Hyperparameter Training Lab":
    st.markdown("### 🧪 Deep Q-Learning Hyperparameter Tuning & Training")
    
    default_p = load_default_params()
    
    c_config, c_charts = st.columns([1, 2])
    
    with c_config:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### ⚙️ Configure Hyperparameters")
        
        lr = st.number_input("Learning Rate (alpha)", 0.0001, 0.01, float(default_p.get("alpha", 0.001)), format="%.4f")
        gamma = st.slider("Discount Factor (gamma)", 0.80, 0.999, float(default_p.get("gamma", 0.99)))
        epsilon_init = st.slider("Initial Epsilon", 0.1, 1.0, float(default_p.get("epsilon_init", 1.0)))
        epsilon_min = st.slider("Minimum Epsilon", 0.01, 0.2, float(default_p.get("epsilon_min", 0.05)))
        epsilon_decay = st.number_input("Epsilon Decay per Episode", 0.990, 0.9999, float(default_p.get("epsilon_decay", 0.9995)), format="%.4f")
        
        replay_size = st.select_slider("Replay Memory Size", options=[10000, 50000, 100000, 200000], value=int(default_p.get("replay_memory_size", 100000)))
        batch_size = st.select_slider("Mini-Batch Size", options=[16, 32, 64, 128], value=int(default_p.get("mini_batch_size", 32)))
        
        episodes_to_run = st.number_input("Number of Episodes to Train", 5, 500, 30)
        
        btn_train = st.button("🚀 Start Training Session", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_charts:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📈 Real-Time Training Analytics")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        reward_chart_placeholder = st.empty()
        epsilon_chart_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    if btn_train:
        env = gym.make("FlappyBird-v0")
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        
        policy_dqn = DQN(num_states, num_actions).to(DEVICE)
        target_dqn = DQN(num_states, num_actions).to(DEVICE)
        target_dqn.load_state_dict(policy_dqn.state_dict())
        
        optimizer = optim.Adam(policy_dqn.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        memory = ReplayMemory(replay_size)
        
        rewards_history = []
        epsilon_history = []
        epsilon = epsilon_init
        steps = 0
        best_reward = float("-inf")
        
        for episode in range(int(episodes_to_run)):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=DEVICE)
            episode_reward = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated) and episode_reward < 1000:
                if random.random() < epsilon:
                    action = env.action_space.sample()
                    action_tensor = torch.tensor(action, dtype=torch.long, device=DEVICE)
                else:
                    with torch.no_grad():
                        action_tensor = policy_dqn(state.unsqueeze(0)).squeeze().argmax()
                        action = action_tensor.item()
                        
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                
                reward_tensor = torch.tensor(reward, dtype=torch.float, device=DEVICE)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float, device=DEVICE)
                
                memory.append((state, action_tensor, next_state_tensor, reward_tensor, terminated))
                steps += 1
                state = next_state_tensor
                
                if len(memory) > batch_size:
                    mini_batch = memory.sample(batch_size)
                    states_b, actions_b, next_states_b, rewards_b, terms_b = zip(*mini_batch)
                    
                    st_stack = torch.stack(states_b)
                    act_stack = torch.stack(actions_b)
                    nxt_stack = torch.stack(next_states_b)
                    rew_stack = torch.stack(rewards_b)
                    term_stack = torch.tensor(terms_b).float().to(DEVICE)
                    
                    with torch.no_grad():
                        target_q = rew_stack + (1 - term_stack) * gamma * target_dqn(nxt_stack).max(dim=1)[0]
                    current_q = policy_dqn(st_stack).gather(dim=1, index=act_stack.unsqueeze(1)).squeeze()
                    
                    loss = loss_fn(current_q, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if steps > 10:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        steps = 0
                        
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            rewards_history.append(episode_reward)
            epsilon_history.append(epsilon)
            
            # Save best checkpoint
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(policy_dqn.state_dict(), DEFAULT_MODEL_PATH)
            
            # Progress & Charts
            progress_bar.progress((episode + 1) / episodes_to_run)
            status_text.write(f"Episode {episode+1}/{episodes_to_run} | Reward: {episode_reward:.1f} | Epsilon: {epsilon:.4f}")
            
            # Reward Plot
            fig_r = px.line(
                x=list(range(1, len(rewards_history) + 1)),
                y=rewards_history,
                labels={"x": "Episode", "y": "Reward"},
                title="Episode Rewards Progress"
            )
            fig_r.update_traces(line_color="#38bdf8", line_width=2)
            fig_r.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#cbd5e1"), height=220)
            reward_chart_placeholder.plotly_chart(fig_r, use_container_width=True)
            
            # Epsilon Plot
            fig_e = px.line(
                x=list(range(1, len(epsilon_history) + 1)),
                y=epsilon_history,
                labels={"x": "Episode", "y": "Epsilon"},
                title="Exploration Rate (Epsilon Decay)"
            )
            fig_e.update_traces(line_color="#c084fc", line_width=2)
            fig_e.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#cbd5e1"), height=200)
            epsilon_chart_placeholder.plotly_chart(fig_e, use_container_width=True)
            
        env.close()
        st.success("🎉 Training Session Completed! Best model saved.")

# ---------------------------------------------------------
# PAGE 3: NEURAL NETWORK INSPECTOR
# ---------------------------------------------------------
elif page == "🧠 Neural Network Inspector":
    st.markdown("### 🧠 PyTorch DQN Model Architecture & Interactive Predictor")
    
    model = get_policy_network(model_path=DEFAULT_MODEL_PATH)
    in_dim = model.model[0].in_features
    
    col_arch, col_sim = st.columns([1, 1.2])
    
    with col_arch:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📐 Neural Network Architecture")
        st.code(f"""
Deep Q-Network (DQN) Architecture:
----------------------------------
Input Layer   : Linear({in_dim} state features -> 256 neurons)
Activation    : ReLU()
Output Layer  : Linear(256 neurons -> 2 Q-values)

Output Actions:
0 -> Do Nothing (Fall)
1 -> Flap Wings (Jump)
""", language="text")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_sim:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🎛️ Interactive State Predictor Simulator")
        st.caption("Adjust key simulated environment features to test predicted Q-values.")
        
        val1 = st.slider("Primary Sensor / Distance Feature 1", -1.0, 1.0, 0.1, 0.05)
        val2 = st.slider("Primary Sensor / Distance Feature 2", -10.0, 10.0, 0.0, 0.5)
        val3 = st.slider("Primary Sensor / Distance Feature 3", 0.0, 2.0, 0.5, 0.05)
        
        # Build tensor of size in_dim matching current model
        sim_state = np.zeros(in_dim, dtype=np.float32)
        if in_dim >= 3:
            sim_state[0] = val1
            sim_state[1] = val2
            sim_state[2] = val3
            
        with torch.no_grad():
            q_out = model(torch.tensor(sim_state, device=DEVICE).unsqueeze(0)).squeeze().cpu().numpy()
            predicted_act = int(np.argmax(q_out))
            
        st.markdown("##### Model Decision Prediction:")
        if predicted_act == 1:
            st.markdown('### <span class="action-badge-flap">RECOMMENDED ACTION: FLAP (1) 🚀</span>', unsafe_allow_html=True)
        else:
            st.markdown('### <span class="action-badge-none">RECOMMENDED ACTION: DO NOTHING (0) 😴</span>', unsafe_allow_html=True)
            
        st.write(f"**Predicted Q(No Flap):** `{q_out[0]:.4f}` | **Predicted Q(Flap):** `{q_out[1]:.4f}`")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE 4: HUMAN PLAY GUIDE
# ---------------------------------------------------------
elif page == "🎮 Human Play Guide":
    st.markdown("### 🎮 Interactive Human Play Mode")
    st.markdown("""
    <div class="glass-card">
        <h4>🕹️ Playing Flappy Bird Manually</h4>
        <p>You can launch the native Pygame window to play Flappy Bird directly with your spacebar!</p>
        <ol>
            <li>Open your terminal in the project directory.</li>
            <li>Run the command: <code>python flappy_bird.py</code></li>
            <li>Press <b>SPACEBAR</b> to flap the bird's wings and pass between the pipes.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
