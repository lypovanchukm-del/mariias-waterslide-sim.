import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- UI SETTINGS ---
st.set_page_config(page_title="Waterslide Safety Analysis", layout="wide")
st.title("Waterslide Engineering: Safety and Physics Simulator")

# --- SIDEBAR PARAMETERS ---
st.sidebar.header("Design Parameters")
# Friction and Physics
mu = st.sidebar.slider("Friction Coefficient (mu)", 0.0, 0.4, 0.05, 0.01)
g = 9.81
a_param = st.sidebar.slider("Curve Steepness (a)", 0.01, 0.1, 0.05)

# Rider Parameters
st.sidebar.header("Rider Parameters")
mass = st.sidebar.slider("Rider Mass (kg)", 30, 120, 75)

# --- PHYSICS ENGINE ---
def get_profile(x):
    """Defines the vertical shape of the slide: y = a * x^2"""
    return a_param * x**2

def get_slope(x):
    """First derivative: y' = 2 * a * x"""
    return 2 * a_param * x

def get_curvature(x):
    """Calculates the radius of curvature: R = (1 + (y')^2)^1.5 / |y''|"""
    y_prime = get_slope(x)
    y_double_prime = 2 * a_param
    radius = (1 + y_prime**2)**1.5 / np.abs(y_double_prime)
    return radius

def physics_engine(t, state):
    """Differential equations for motion on a curved surface"""
    x, v = state
    slope = get_slope(x)
    angle = np.arctan(slope)
    
    # Forces calculation
    # friction = mu * N, where N = m * g * cos(theta)
    friction = mu * g * np.cos(angle) * np.sign(v)
    dvdt = -g * np.sin(angle) - friction
    dxdt = v * np.cos(angle)
    
    return [dxdt, dvdt]

# --- SIMULATION EXECUTION ---
t_span = (0, 15)
t_eval = np.linspace(0, 15, 500)
# Start at x = -10 (left side of the parabola), velocity = 0
sol = solve_ivp(physics_engine, t_span, [-10, 0], t_eval=t_eval)

x_res = sol.y[0]
v_res = sol.y[1]

# --- SAFETY CALCULATIONS ---
radii = get_curvature(x_res)
# Centripetal acceleration
centripetal_acc = v_res**2 / radii
# Normal force in G-units: (g*cos(angle) + a_cent) / g
g_force = (g * np.cos(np.arctan(get_slope(x_res))) + centripetal_acc) / g
# Pressure Force in Newtons
pressure_force = mass * g_force * g

# --- VISUALIZATION ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Trajectory Visualization")
    fig_track, ax_track = plt.subplots()
    x_track = np.linspace(-12, 12, 500)
    ax_track.plot(x_track, get_profile(x_track), 'k--', label="Slide Profile")
    ax_track.plot(x_res, get_profile(x_res), 'r-', linewidth=2, label="Rider Path")
    ax_track.set_xlabel("Horizontal Distance (m)")
    ax_track.set_ylabel("Vertical Height (m)")
    ax_track.legend()
    st.pyplot(fig_track)

with col2:
    st.subheader("G-Force Analysis")
    fig_g, ax_g = plt.subplots()
    ax_g.plot(sol.t, g_force, color='orange')
    ax_g.axhline(y=3.0, color='r', linestyle='--', label="Danger Threshold (3.0G)")
    ax_g.set_xlabel("Time (s)")
    ax_g.set_ylabel("G-Force (units of g)")
    ax_g.legend()
    st.pyplot(fig_g)

# --- RESULTS SUMMARY ---
st.divider()
max_g = np.max(g_force)
max_v = np.max(np.abs(v_res))
max_p = np.max(pressure_force)

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Max Velocity", f"{max_v:.2f} m/s")
col_m2.metric("Max G-Force", f"{max_g:.2f} G")
col_m3.metric("Peak Pressure", f"{max_p:.0f} N")

if max_g > 3.0:
    st.error("SAFETY ALERT: Excessive G-Force. Structural or physical risk.")
else:
    st.success("Analysis: Parameters are within safe operational limits.")
st.header("Safety Analysis Report")

if max_g > 3.0:
    st.error(f"DANGER: Maximum G-force reached {max_g:.2f} G. This exceeds safety limits!")
    st.write("Recommendation: Decrease the 'Curve Steepness' or increase 'Friction'.")
else:
    st.success(f"SAFE: Maximum G-force is {max_g:.2f} G. Design is optimal.")

st.info(f"Rider will reach the bottom at a speed of {max_v:.1f} m/s.")