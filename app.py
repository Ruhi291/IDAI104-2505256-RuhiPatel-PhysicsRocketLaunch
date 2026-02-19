import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚀 Rocket Launch Path Visualization",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── SESSION STATE ───────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ─── THEME ───────────────────────────────────────────────────────────────────────
def apply_theme():
    if st.session_state.dark_mode:
        bg = "#0d1117"
        card_bg = "#161b22"
        text = "#e6edf3"
        accent = "#58a6ff"
        sub = "#8b949e"
        plot_bg = "#0d1117"
        grid = "#21262d"
    else:
        bg = "#f0f4f8"
        card_bg = "#ffffff"
        text = "#1c1e21"
        accent = "#1a73e8"
        sub = "#555e68"
        plot_bg = "#ffffff"
        grid = "#e0e4ea"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] {{
        background-color: {bg} !important;
        color: {text} !important;
        font-family: 'Space Mono', monospace;
    }}
    .stApp {{
        background-color: {bg} !important;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {card_bg} !important;
        border-right: 1px solid {grid};
    }}
    .block-container {{
        padding: 2rem 3rem;
    }}
    h1, h2, h3 {{
        font-family: 'Orbitron', sans-serif !important;
        color: {accent} !important;
        letter-spacing: 0.05em;
    }}
    .metric-card {{
        background: {card_bg};
        border: 1px solid {grid};
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }}
    .stButton>button {{
        background: linear-gradient(135deg, {accent}, #f78166) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
        padding: 0.6rem 2rem !important;
        transition: all 0.3s ease !important;
    }}
    .stButton>button:hover {{
        transform: scale(1.04) !important;
        box-shadow: 0 0 20px {accent}66 !important;
    }}
    .section-header {{
        font-family: 'Orbitron', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: {accent};
        border-bottom: 2px solid {accent}44;
        padding-bottom: 0.4rem;
        margin: 2rem 0 1rem 0;
    }}
    .insight-box {{
        background: {card_bg};
        border-left: 4px solid {accent};
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        color: {text};
    }}
    label, .stSlider label, .stSelectbox label {{
        color: {sub} !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.05em !important;
    }}
    </style>
    """, unsafe_allow_html=True)
    return bg, card_bg, text, accent, sub, plot_bg, grid

bg, card_bg, text_col, accent, sub, plot_bg, grid_col = apply_theme()

# ─── SYNTHETIC DATA GENERATION ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 300

    mission_types = ["Lunar", "Mars", "ISS Supply", "Satellite Deployment", "Deep Space"]
    vehicles = ["Falcon 9", "SLS", "Starship", "Atlas V", "Ariane 5"]
    success_options = ["Success", "Failure"]

    dates = pd.date_range(start="2000-01-01", end="2024-12-31", periods=n)

    df = pd.DataFrame({
        "Launch Date": dates,
        "Mission Type": np.random.choice(mission_types, n),
        "Launch Vehicle": np.random.choice(vehicles, n),
        "Payload Weight (kg)": np.random.uniform(500, 20000, n).round(1),
        "Fuel Consumption (tonnes)": np.random.uniform(100, 800, n).round(2),
        "Mission Cost (M$)": np.random.uniform(50, 2000, n).round(1),
        "Mission Success": np.random.choice(success_options, n, p=[0.78, 0.22]),
        "Mission Duration (days)": np.random.uniform(1, 900, n).round(1),
        "Distance from Earth (km)": np.random.uniform(400, 5e8, n).round(0),
        "Crew Size": np.random.choice([0, 2, 4, 6, 7], n),
        "Scientific Yield (units)": np.random.uniform(0, 100, n).round(2),
    })

    # Introduce some realistic missing values & duplicates for cleaning demo
    for col in ["Fuel Consumption (tonnes)", "Scientific Yield (units)", "Mission Cost (M$)"]:
        idx = np.random.choice(df.index, size=int(0.04 * n), replace=False)
        df.loc[idx, col] = np.nan

    df = pd.concat([df, df.sample(10, random_state=1)], ignore_index=True)

    # ── Cleaning ────────────────────────────────────────────────
    df["Launch Date"] = pd.to_datetime(df["Launch Date"])
    numeric_cols = [
        "Payload Weight (kg)", "Fuel Consumption (tonnes)", "Mission Cost (M$)",
        "Mission Duration (days)", "Distance from Earth (km)", "Crew Size", "Scientific Yield (units)"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.drop_duplicates(inplace=True)
    df.dropna(subset=["Launch Date", "Mission Type", "Launch Vehicle"], inplace=True)

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    df["Year"] = df["Launch Date"].dt.year
    return df

df_raw = load_data()

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='font-family:Orbitron;'>⚙️ Controls</h2>", unsafe_allow_html=True)

    # Dark mode toggle
    mode_label = "☀️ Light Mode" if st.session_state.dark_mode else "🌙 Dark Mode"
    if st.button(mode_label):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown("---")
    st.markdown("**🔭 Mission Filters**")

    mission_options = ["All"] + sorted(df_raw["Mission Type"].unique().tolist())
    selected_mission = st.selectbox("Mission Type", mission_options)

    vehicle_options = ["All"] + sorted(df_raw["Launch Vehicle"].unique().tolist())
    selected_vehicle = st.selectbox("Launch Vehicle", vehicle_options)

    dist_min = float(df_raw["Distance from Earth (km)"].min())
    dist_max = float(df_raw["Distance from Earth (km)"].max())
    dist_range = st.slider(
        "Distance from Earth (km)",
        min_value=dist_min,
        max_value=dist_max,
        value=(dist_min, dist_max),
        format="%.0f"
    )

    year_min = int(df_raw["Year"].min())
    year_max = int(df_raw["Year"].max())
    year_range = st.slider("Year Range", year_min, year_max, (year_min, year_max))

    st.markdown("---")
    st.markdown("**🚀 Simulation Parameters**")
    thrust = st.slider("Thrust (kN)", 500, 10000, 3000, 100)
    initial_mass = st.slider("Initial Mass (kg)", 50000, 500000, 200000, 5000)
    fuel_mass = st.slider("Fuel Mass (kg)", 10000, 300000, 100000, 5000)
    drag_coeff = st.slider("Drag Coefficient", 0.1, 2.0, 0.5, 0.05)

# ─── FILTER DATA ─────────────────────────────────────────────────────────────────
df = df_raw.copy()
if selected_mission != "All":
    df = df[df["Mission Type"] == selected_mission]
if selected_vehicle != "All":
    df = df[df["Launch Vehicle"] == selected_vehicle]
df = df[
    (df["Distance from Earth (km)"] >= dist_range[0]) &
    (df["Distance from Earth (km)"] <= dist_range[1])
]
df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]

# ─── MATPLOTLIB THEME HELPER ─────────────────────────────────────────────────────
def mpl_style():
    if st.session_state.dark_mode:
        plt.style.use("dark_background")
        spine_col = "#21262d"
        txt_col = "#e6edf3"
        face = "#161b22"
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
        spine_col = "#e0e4ea"
        txt_col = "#1c1e21"
        face = "#ffffff"
    return spine_col, txt_col, face

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 – PROJECT OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("<h1>🚀 Rocket Launch Path Visualization</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#8b949e;font-family:Space Mono;'>Mathematics for AI · Scenario 1</p>", unsafe_allow_html=True)

with st.expander("📖 Project Overview — The Physics Behind Rocket Flight", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class='insight-box'>
        <b>Newton's Second Law</b><br>
        <code>F = m × a</code><br><br>
        The net force on a rocket equals its mass times acceleration.
        As fuel burns, mass decreases and acceleration increases.
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='insight-box'>
        <b>Thrust & Gravity</b><br>
        <code>F_net = T − W − D</code><br><br>
        Thrust must overcome gravitational pull (mg) and atmospheric
        drag to achieve liftoff and acceleration.
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='insight-box'>
        <b>Drag Force</b><br>
        <code>D = Cd × v²</code><br><br>
        Drag grows with the square of velocity. A lower drag coefficient
        means less air resistance and higher peak altitude.
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 – DATA OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>📊 Data Overview</div>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
metrics = [
    ("Total Missions", len(df), ""),
    ("Success Rate", f"{(df['Mission Success']=='Success').mean()*100:.1f}%", ""),
    ("Avg Mission Cost", f"${df['Mission Cost (M$)'].mean():.0f}M", ""),
    ("Avg Payload", f"{df['Payload Weight (kg)'].mean():.0f} kg", ""),
]
for col, (label, val, _) in zip([col1, col2, col3, col4], metrics):
    with col:
        st.metric(label, val)

with st.expander("🔍 View Filtered Dataset"):
    st.dataframe(df.reset_index(drop=True), use_container_width=True, height=300)
    st.caption(f"Showing {len(df)} records after filters applied.")

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 – MISSION ANALYSIS DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>🛰️ Mission Analysis Dashboard</div>", unsafe_allow_html=True)

# Guard: need at least some data
if len(df) < 3:
    st.warning("Not enough data with current filters. Please widen your filter selection.")
else:
    spine_col, txt_col, face_col = mpl_style()

    # ── Row 1: Scatter + Bar ─────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Payload Weight vs Fuel Consumption**")
        fig1 = px.scatter(
            df,
            x="Payload Weight (kg)",
            y="Fuel Consumption (tonnes)",
            color="Mission Success",
            symbol="Mission Type",
            color_discrete_map={"Success": "#3fb950", "Failure": "#f85149"},
            template="plotly_dark" if st.session_state.dark_mode else "plotly_white",
            hover_data=["Launch Vehicle", "Year"],
        )
        fig1.update_layout(
            paper_bgcolor=plot_bg,
            plot_bgcolor=plot_bg,
            font_color=txt_col,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("**Mission Cost: Success vs Failure**")
        cost_grp = df.groupby("Mission Success")["Mission Cost (M$)"].mean().reset_index()
        fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor=face_col)
        ax2.set_facecolor(face_col)
        colors = ["#3fb950", "#f85149"]
        bars = ax2.bar(cost_grp["Mission Success"], cost_grp["Mission Cost (M$)"], color=colors, width=0.5, edgecolor="none")
        ax2.set_xlabel("Mission Outcome", color=txt_col, fontsize=9)
        ax2.set_ylabel("Avg Cost (M$)", color=txt_col, fontsize=9)
        ax2.tick_params(colors=txt_col)
        for spine in ax2.spines.values():
            spine.set_edgecolor(spine_col)
        for bar, v in zip(bars, cost_grp["Mission Cost (M$)"]):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f"${v:.0f}M", ha="center", color=txt_col, fontsize=9)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ── Row 2: Line + Box ────────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Mission Duration vs Distance from Earth**")
        df_sorted = df.sort_values("Distance from Earth (km)")
        fig3 = px.line(
            df_sorted,
            x="Distance from Earth (km)",
            y="Mission Duration (days)",
            color="Mission Type",
            template="plotly_dark" if st.session_state.dark_mode else "plotly_white",
            line_shape="spline",
        )
        fig3.update_layout(
            paper_bgcolor=plot_bg,
            plot_bgcolor=plot_bg,
            font_color=txt_col,
            legend=dict(orientation="h", yanchor="bottom", y=-0.35, font_size=9),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("**Crew Size vs Mission Success**")
        fig4, ax4 = plt.subplots(figsize=(6, 4), facecolor=face_col)
        ax4.set_facecolor(face_col)
        palette = {"Success": "#3fb950", "Failure": "#f85149"}
        sns.boxplot(
            data=df,
            x="Mission Success",
            y="Crew Size",
            palette=palette,
            ax=ax4,
            linewidth=1.2,
            flierprops=dict(marker="o", markersize=4, linestyle="none"),
        )
        ax4.tick_params(colors=txt_col)
        ax4.set_xlabel("Mission Outcome", color=txt_col, fontsize=9)
        ax4.set_ylabel("Crew Size", color=txt_col, fontsize=9)
        for spine in ax4.spines.values():
            spine.set_edgecolor(spine_col)
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    # ── Row 3: Scatter + Heatmap ──────────────────────────────────────────────
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("**Scientific Yield vs Mission Cost**")
        fig5, ax5 = plt.subplots(figsize=(6, 4), facecolor=face_col)
        ax5.set_facecolor(face_col)
        success_mask = df["Mission Success"] == "Success"
        ax5.scatter(
            df.loc[success_mask, "Mission Cost (M$)"],
            df.loc[success_mask, "Scientific Yield (units)"],
            alpha=0.6, c="#3fb950", label="Success", s=30, edgecolors="none"
        )
        ax5.scatter(
            df.loc[~success_mask, "Mission Cost (M$)"],
            df.loc[~success_mask, "Scientific Yield (units)"],
            alpha=0.6, c="#f85149", label="Failure", s=30, edgecolors="none"
        )
        ax5.set_xlabel("Mission Cost (M$)", color=txt_col, fontsize=9)
        ax5.set_ylabel("Scientific Yield (units)", color=txt_col, fontsize=9)
        ax5.tick_params(colors=txt_col)
        ax5.legend(fontsize=8, facecolor=face_col, labelcolor=txt_col, framealpha=0.5)
        for spine in ax5.spines.values():
            spine.set_edgecolor(spine_col)
        fig5.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

    with col6:
        st.markdown("**Correlation Heatmap (Numeric Columns)**")
        numeric_df = df.select_dtypes(include=np.number).drop(columns=["Year"], errors="ignore")
        corr = numeric_df.corr()
        fig6, ax6 = plt.subplots(figsize=(6, 4), facecolor=face_col)
        ax6.set_facecolor(face_col)
        cmap = "RdYlGn" if not st.session_state.dark_mode else "coolwarm"
        sns.heatmap(
            corr,
            ax=ax6,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            linewidths=0.4,
            linecolor=spine_col,
            annot_kws={"size": 7},
            cbar_kws={"shrink": 0.8},
        )
        ax6.tick_params(colors=txt_col, labelsize=7)
        plt.setp(ax6.get_xticklabels(), rotation=30, ha="right", color=txt_col)
        plt.setp(ax6.get_yticklabels(), color=txt_col)
        fig6.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 – ROCKET SIMULATION
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>🔥 Rocket Launch Simulation</div>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:{sub};font-size:0.85rem;'>Configure thrust, mass, and drag in the sidebar, then launch.</p>",
    unsafe_allow_html=True
)

col_params, col_chart = st.columns([1, 2])

with col_params:
    st.markdown(f"""<div class='insight-box'>
    <b>Parameters</b><br>
    Thrust: <code>{thrust:,} kN</code><br>
    Initial Mass: <code>{initial_mass:,} kg</code><br>
    Fuel Mass: <code>{fuel_mass:,} kg</code><br>
    Drag Coeff: <code>{drag_coeff}</code>
    </div>""", unsafe_allow_html=True)
    launch_btn = st.button("🚀 Launch Simulation")

with col_chart:
    if launch_btn:
        # Simulation parameters
        dt = 1.0           # time step (s)
        steps = 200
        g = 9.81           # gravity (m/s²)
        thrust_N = thrust * 1000  # kN → N
        burn_rate = fuel_mass / steps  # kg/s

        mass = float(initial_mass)
        velocity = 0.0
        altitude = 0.0

        times, altitudes, velocities, accelerations, masses = [], [], [], [], []

        for i in range(steps):
            drag = drag_coeff * velocity ** 2
            net_force = thrust_N - (mass * g) - drag
            accel = net_force / mass if mass > 0 else 0.0

            velocity += accel * dt
            velocity = max(velocity, 0.0)  # can't go negative (simplification)
            altitude += velocity * dt

            times.append(i * dt)
            altitudes.append(altitude / 1000)  # convert to km
            velocities.append(velocity)
            accelerations.append(accel)
            masses.append(mass)

            # Burn fuel
            if mass > (initial_mass - fuel_mass):
                mass -= burn_rate
                mass = max(mass, initial_mass - fuel_mass)

        sim_df = pd.DataFrame({
            "Time (s)": times,
            "Altitude (km)": altitudes,
            "Velocity (m/s)": velocities,
            "Acceleration (m/s²)": accelerations,
            "Mass (kg)": masses,
        })

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=sim_df["Time (s)"],
            y=sim_df["Altitude (km)"],
            mode="lines",
            name="Altitude (km)",
            line=dict(color="#58a6ff", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.1)"
        ))
        fig_sim.add_trace(go.Scatter(
            x=sim_df["Time (s)"],
            y=sim_df["Velocity (m/s)"] / 1000,
            mode="lines",
            name="Velocity (km/s)",
            line=dict(color="#f78166", width=1.5, dash="dot"),
            yaxis="y2"
        ))
        fig_sim.update_layout(
            template="plotly_dark" if st.session_state.dark_mode else "plotly_white",
            paper_bgcolor=plot_bg,
            plot_bgcolor=plot_bg,
            font_color=txt_col,
            xaxis_title="Time (s)",
            yaxis_title="Altitude (km)",
            yaxis2=dict(title="Velocity (km/s)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.08),
            margin=dict(l=20, r=20, t=50, b=20),
            title="Altitude & Velocity vs Time",
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        peak_alt = sim_df["Altitude (km)"].max()
        peak_vel = sim_df["Velocity (m/s)"].max()
        st.success(f"Peak Altitude: **{peak_alt:.1f} km** | Peak Velocity: **{peak_vel:.0f} m/s**")
    else:
        st.info("👈 Set simulation parameters in the sidebar and click **Launch Simulation**.")

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 – KEY INSIGHTS
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>💡 Key Insights</div>", unsafe_allow_html=True)

if len(df) >= 3:
    success_rate = (df["Mission Success"] == "Success").mean() * 100
    avg_cost_success = df.loc[df["Mission Success"] == "Success", "Mission Cost (M$)"].mean()
    avg_cost_fail = df.loc[df["Mission Success"] == "Failure", "Mission Cost (M$)"].mean()
    top_vehicle = df["Launch Vehicle"].value_counts().idxmax()
    corr_pay_fuel = df["Payload Weight (kg)"].corr(df["Fuel Consumption (tonnes)"])

    ins = [
        f"✅ **{success_rate:.1f}%** of filtered missions succeeded.",
        f"💰 Successful missions cost on average **${avg_cost_success:.0f}M** vs **${avg_cost_fail:.0f}M** for failures.",
        f"🚀 Most used launch vehicle in selection: **{top_vehicle}**.",
        f"📦 Correlation between Payload Weight & Fuel Consumption: **{corr_pay_fuel:.2f}** — {'strong positive' if corr_pay_fuel > 0.5 else 'moderate/weak'} relationship.",
        "🌌 Missions to greater distances require significantly longer durations and higher fuel loads.",
        "⚙️ In simulation: lower drag coefficients and higher thrust dramatically increase peak altitude and velocity.",
    ]
    for insight in ins:
        st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)
else:
    st.warning("Apply less restrictive filters to see insights.")

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("🚀 Rocket Launch Path Visualization · Mathematics for AI · Built with Streamlit")

