import streamlit as st
import numpy as np
import rasterio
import tensorflow as tf
import math, random
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from collections import defaultdict

# ---------------- Streamlit UI ----------------
st.title("🔥 SpreadNet Fire Spread Simulation")

# File uploaders
slope_file = st.file_uploader("Upload Slope Raster (.tif)", type="tif")
fuel_file  = st.file_uploader("Upload Fuel Raster (.tif)", type="tif")

# Simulation parameters
MOIST_GLOBAL = st.slider("Global Moisture (%)", min_value=0.0, max_value=40.0, value=1.0)
WIND_SPEED   = st.slider("Wind Speed (m/s)", min_value=0.0, max_value=30.0, value=10.0)
WIND_DIR_DEG = st.slider("Wind Direction (° from N)", min_value=0, max_value=359, value=250)
STEP_MIN     = st.slider("Time Step (minutes)", min_value=1, max_value=60, value=10)
MAX_SIM_MIN  = st.slider("Max Simulation Time (minutes)", min_value=10, max_value=3000, value=2240)
run_button   = st.button("Run SpreadNet Simulation")

# ---------- constants and model setup ----------
VALID_FUELS = [1,2,3,4,5,6,7,8,9,10,11,12,13,98,93]
FUEL_EMB = defaultdict(lambda:[0,0,0], {
    1:[1,0,0],2:[1,0,0],3:[1,0,0],
    4:[0,1,0],5:[0,1,0],6:[0,1,0],
    7:[0,0,1],8:[0,0,1],9:[0,0,1],
    10:[.5,.5,0],11:[0,.5,.5],12:[.5,0,.5],13:[.7,.3,0]
})
NEIGH = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def build_spreadnet():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1,  activation='sigmoid')
    ])

def wind_align_deg(dir_from, dir_to):
    return math.cos(math.radians(dir_from - dir_to))

def direction_deg(y,x,ny,nx):
    return math.degrees(math.atan2(nx-x, ny-y)) % 360

def predict_prob_batch(batch_feats):
    return net(np.asarray(batch_feats, 'float32'), training=False).numpy().ravel()

def load_raster(file, mask_zero=False):
    with rasterio.open(file) as src:
        arr = src.read(1, masked=True)
        data = np.nan_to_num(arr.filled(0))
        if mask_zero: data[data==0] = np.nan
        return data, src.transform, src.shape

# ---------- Simulation ----------
if run_button and slope_file and fuel_file:
    slope, transform, (rows, cols) = load_raster(slope_file)
    fuel , _        , _            = load_raster(fuel_file)

    if slope.max() > 90:
        slope = np.degrees(np.arctan(slope / 100))
    slope = np.clip(slope, 0, 60)

    CELL = transform.a
    DIAG = CELL * math.sqrt(2)

    net = build_spreadnet()
    net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    X_dummy = np.random.rand(100, 8).astype('float32')
    y_dummy = np.random.randint(0, 2, 100).astype('float32')
    net.fit(X_dummy, y_dummy, epochs=1, batch_size=32, verbose=0)  # dummy train to initialize

    burn = np.zeros((rows, cols), np.int8)
    burn[rows//2, cols//2] = 1
    minutes = 0
    runs = []

    while burn.any() and minutes < MAX_SIM_MIN:
        new = burn.copy()
        feats = []
        cells = []

        for y,x in zip(*np.where(burn == 1)):
            new[y,x] = 2
            for dy,dx in NEIGH:
                ny,nx = y+dy, x+dx
                if not (0 <= ny < rows and 0 <= nx < cols): continue
                if burn[ny,nx] != 0: continue
                if fuel[ny,nx] in [93, 98, 99]: continue

                emb = FUEL_EMB[int(fuel[ny,nx])]
                feat = emb + [
                    slope[ny,nx]/60,
                    MOIST_GLOBAL/40,
                    WIND_SPEED/30,
                    wind_align_deg(WIND_DIR_DEG, direction_deg(y,x,ny,nx)),
                    (DIAG if dy*dx else CELL)/DIAG
                ]
                feats.append(feat)
                cells.append((ny,nx))

        if feats:
            probs = predict_prob_batch(feats)
            for (ny,nx), p in zip(cells, probs):
                if random.random() < p:
                    new[ny,nx] = 1

        burn = new
        minutes += STEP_MIN
        runs.append((minutes, burn.copy()))

    arrival = np.full((rows, cols), np.nan)
    for i, (m, b) in enumerate(runs):
        arrival[(b==2) & np.isnan(arrival)] = i+1

    # ---------- Plot ----------
    xmin, xmax = transform.c, transform.c + transform.a * cols
    ymin, ymax = transform.f + transform.e * rows, transform.f
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(fuel, cmap='gray_r', extent=[xmin, xmax, ymin, ymax], origin='upper')
    im = ax.imshow(arrival, cmap=get_cmap('plasma', len(runs)),
                   extent=[xmin, xmax, ymin, ymax], origin='upper',
                   vmin=1, vmax=len(runs), alpha=0.75)
    ax.set_title("Fire Arrival Time (minutes)")
    ax.axis('off')
    cbar = fig.colorbar(im, ax=ax, ticks=[1, len(runs)])
    cbar.ax.set_yticklabels([f"{runs[0][0]} min", f"{runs[-1][0]} min"])
    st.pyplot(fig)
