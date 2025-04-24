import streamlit as st
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap, BoundaryNorm
from collections import defaultdict

# --- Sidebar: User presets ---
st.title("Interactive Fire Spread Simulation")
st.sidebar.header("Simulation Settings")

# Default GitHub URLs for rasters
DEFAULT_SLOPE_URL = "https://raw.githubusercontent.com/yourusername/yourrepo/main/LC20_SlpD_220_SMALL2.tif"
DEFAULT_FUEL_URL  = "https://raw.githubusercontent.com/yourusername/yourrepo/main/LC22_F13_230_SMALL2.tif"

slope_url = st.sidebar.text_input("Slope raster URL", DEFAULT_SLOPE_URL)
fuel_url  = st.sidebar.text_input("Fuel model raster URL", DEFAULT_FUEL_URL)

WIND_SPEED_CONST   = st.sidebar.slider("Wind speed (m/s)", 0.0, 30.0, 10.0)
WIND_DIR_CONST     = st.sidebar.slider("Wind direction (°)", 0, 360, 210)
IGNITION_DIAMOND   = st.sidebar.checkbox("Use diamond ignition area?", False)
DEAD_FUEL_MOISTURE = st.sidebar.slider("Dead fuel moisture (%)", 0.0, 40.0, 2.0)
BASE_ROS           = st.sidebar.number_input("Base rate of spread", 0.0, 0.01, 0.00001, format="%.5f")
TIMESTEP_MIN       = st.sidebar.slider("Time step (minutes)", 1, 60, 10)
MAX_MINUTES        = st.sidebar.slider("Max simulation time (minutes)", 60, 14400, 2400)

st.write("---")
st.write("## Simulation Overview")
st.write("This app simulates how a fire spreads across a landscape. You can adjust wind speed, direction, moisture, and other settings. The background rasters represent slope and fuel types.")

# --- Load rasters ---
@st.cache_data
def load_raster(url):
    with rasterio.open(url) as src:
        data = src.read(1, masked=True) if 'LC20' in url else src.read(1)
        transform = src.transform
        shape = src.shape
    return data, transform, shape

slope, _, _ = load_raster(slope_url)
fuelmodel, transform, (rows, cols) = load_raster(fuel_url)

# Convert masked slope to array
slope = np.nan_to_num(slope.filled(0))

# --- Prepare simulation ---
CELL     = transform.a
diag_len = np.sqrt(2) * CELL

# Convert slope percent to degrees if needed
if slope.max() > 90:
    slope = np.arctan(slope / 100) * 180 / np.pi
slope = np.clip(slope, 0, 60)

FUEL_ROS = defaultdict(lambda: 0.1, {1:1.5,2:1.2,3:2.0,4:0.8,5:0.6,6:0.5,7:0.4,8:0.3,9:0.35,10:0.7,11:0.9,12:1.1,13:1.4})
neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

# Ignition setup
ignition = np.zeros((rows, cols), dtype=np.uint8)
cy, cx = rows//2, cols//2
if IGNITION_DIAMOND:
    for dy in range(-2,3):
        for dx in range(-2+abs(dy),3-abs(dy)):
            ignition[cy+dy, cx+dx] = 1
else:
    ignition[cy, cx] = 1

# Spread probability function
def spread_prob(y, x, ny, nx):
    fm = int(fuelmodel[ny, nx])
    fuel_mult  = FUEL_ROS[fm]
    moist_mult = max(0, 1 - DEAD_FUEL_MOISTURE/40)
    beta       = np.deg2rad(slope[ny, nx])
    slope_mult = min(1 + 5.275*np.tan(beta)**2, 10)
    ws = WIND_SPEED_CONST
    wd = np.deg2rad(WIND_DIR_CONST)
    wind_vec = np.array([np.cos(wd), np.sin(wd)])
    dir_vec  = np.array([ny-y, nx-x]) / np.hypot(ny-y, nx-x)
    wind_align = np.dot(wind_vec, dir_vec)
    wind_mult  = np.exp(0.3*ws*wind_align)
    ros = BASE_ROS * fuel_mult * moist_mult * slope_mult * wind_mult
    dist = diag_len if (ny-y)*(nx-x) != 0 else CELL
    prob = 1 - np.exp(-ros*TIMESTEP_MIN/dist)
    return np.clip(prob, 0, 1)

# Run simulation
burn = np.zeros((rows, cols), dtype=np.int8)
burn[ignition==1] = 1
minutes = 0
runs = []
while np.any(burn==1) and minutes < MAX_MINUTES:
    new = burn.copy()
    for y, x in zip(*np.where(burn==1)):
        new[y, x] = 2
        for dy, dx in neighbors:
            ny, nx = y+dy, x+dx
            if 0<=ny<rows and 0<=nx<cols and burn[ny, nx]==0:
                if np.random.rand() < spread_prob(y, x, ny, nx):
                    new[ny, nx] = 1
    burn = new
    minutes += TIMESTEP_MIN
    if minutes % TIMESTEP_MIN == 0:
        runs.append((minutes, burn.copy()))

# Build arrival map
arrival_map = np.full((rows, cols), np.nan)
for i, (mins, b) in enumerate(runs):
    mask = (b==2)
    arrival_map[mask & np.isnan(arrival_map)] = i+1

# Compute extent with 5% buffer
x_min = transform.c
x_max = transform.c + transform.a * cols
y_max = transform.f
y_min = transform.f + transform.e * rows
x_buf = (x_max - x_min) * 0.05
y_buf = (y_max - y_min) * 0.05
extent = (x_min - x_buf, x_max + x_buf, y_min - y_buf, y_max + y_buf)

# Plot results
st.write("## Fire Arrival Time over Fuel Model")
fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(fuelmodel, cmap="gray_r", origin="upper", extent=extent)
cmap = get_cmap("plasma", len(runs))
im = ax.imshow(arrival_map, cmap=cmap, vmin=1, vmax=len(runs), origin="upper", alpha=0.75, extent=extent)
ax.axis('off')
cbar = fig.colorbar(im, ax=ax, ticks=np.arange(1, len(runs)+1))
cbar.ax.set_yticklabels([f"{m} min" for m, _ in runs])
cbar.set_label("Fire arrival time")
st.pyplot(fig)

st.write("---")
st.write("## Individual Time Steps")
burn_cmap = ListedColormap(["white","orange","red"])
burn_norm = BoundaryNorm([-0.5,0.5,1.5,2.5], burn_cmap.N)
cols_sub = min(len(runs), 3)
rows_sub = (len(runs) + cols_sub - 1) // cols_sub
fig2, axes = plt.subplots(rows_sub, cols_sub, figsize=(12, 4*rows_sub))
for idx, (mins, b) in enumerate(runs):
    ax = axes.flatten()[idx]
    ax.imshow(b, cmap=burn_cmap, norm=burn_norm, origin="upper")
    ax.set_title(f"{mins} min")
    ax.axis('off')
st.pyplot(fig2)
