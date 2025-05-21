# import streamlit as st
# import numpy as np
# import rasterio
# import tensorflow as tf
# import math, random
# import matplotlib.pyplot as plt
# from matplotlib.cm import get_cmap
# from collections import defaultdict

# # ---------------- Streamlit UI ----------------
# st.title("🔥 SpreadNet Fire Spread Simulation")

# # File uploaders
# slope_file = st.file_uploader("Upload Slope Raster (.tif)", type="tif")
# fuel_file  = st.file_uploader("Upload Fuel Raster (.tif)", type="tif")

# # Simulation parameters
# MOIST_GLOBAL = st.slider("Global Moisture (%)", min_value=0.0, max_value=40.0, value=1.0)
# WIND_SPEED   = st.slider("Wind Speed (m/s)", min_value=0.0, max_value=30.0, value=10.0)
# WIND_DIR_DEG = st.slider("Wind Direction (° from N)", min_value=0, max_value=359, value=250)
# STEP_MIN     = st.slider("Time Step (minutes)", min_value=1, max_value=60, value=10)
# MAX_SIM_MIN  = st.slider("Max Simulation Time (minutes)", min_value=10, max_value=3000, value=2240)
# run_button   = st.button("Run SpreadNet Simulation")

# # ---------- constants and model setup ----------
# VALID_FUELS = [1,2,3,4,5,6,7,8,9,10,11,12,13,98,93]
# FUEL_EMB = defaultdict(lambda:[0,0,0], {
#     1:[1,0,0],2:[1,0,0],3:[1,0,0],
#     4:[0,1,0],5:[0,1,0],6:[0,1,0],
#     7:[0,0,1],8:[0,0,1],9:[0,0,1],
#     10:[.5,.5,0],11:[0,.5,.5],12:[.5,0,.5],13:[.7,.3,0]
# })
# NEIGH = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

# def build_spreadnet():
#     return tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(8,)),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dense(16, activation='relu'),
#         tf.keras.layers.Dense(1,  activation='sigmoid')
#     ])

# def wind_align_deg(dir_from, dir_to):
#     return math.cos(math.radians(dir_from - dir_to))

# def direction_deg(y,x,ny,nx):
#     return math.degrees(math.atan2(nx-x, ny-y)) % 360

# def predict_prob_batch(batch_feats):
#     return net(np.asarray(batch_feats, 'float32'), training=False).numpy().ravel()

# def load_raster(file, mask_zero=False):
#     with rasterio.open(file) as src:
#         arr = src.read(1, masked=True)
#         data = np.nan_to_num(arr.filled(0))
#         if mask_zero: data[data==0] = np.nan
#         return data, src.transform, src.shape

# # ---------- Simulation ----------
# if run_button and slope_file and fuel_file:
#     slope, transform, (rows, cols) = load_raster(slope_file)
#     fuel , _        , _            = load_raster(fuel_file)

#     if slope.max() > 90:
#         slope = np.degrees(np.arctan(slope / 100))
#     slope = np.clip(slope, 0, 60)

#     CELL = transform.a
#     DIAG = CELL * math.sqrt(2)

#     net = build_spreadnet()
#     net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     X_dummy = np.random.rand(100, 8).astype('float32')
#     y_dummy = np.random.randint(0, 2, 100).astype('float32')
#     net.fit(X_dummy, y_dummy, epochs=1, batch_size=32, verbose=0)  # dummy train to initialize

#     burn = np.zeros((rows, cols), np.int8)
#     burn[rows//2, cols//2] = 1
#     minutes = 0
#     runs = []

#     while burn.any() and minutes < MAX_SIM_MIN:
#         new = burn.copy()
#         feats = []
#         cells = []

#         for y,x in zip(*np.where(burn == 1)):
#             new[y,x] = 2
#             for dy,dx in NEIGH:
#                 ny,nx = y+dy, x+dx
#                 if not (0 <= ny < rows and 0 <= nx < cols): continue
#                 if burn[ny,nx] != 0: continue
#                 if fuel[ny,nx] in [93, 98, 99]: continue

#                 emb = FUEL_EMB[int(fuel[ny,nx])]
#                 feat = emb + [
#                     slope[ny,nx]/60,
#                     MOIST_GLOBAL/40,
#                     WIND_SPEED/30,
#                     wind_align_deg(WIND_DIR_DEG, direction_deg(y,x,ny,nx)),
#                     (DIAG if dy*dx else CELL)/DIAG
#                 ]
#                 feats.append(feat)
#                 cells.append((ny,nx))

#         if feats:
#             probs = predict_prob_batch(feats)
#             for (ny,nx), p in zip(cells, probs):
#                 if random.random() < p:
#                     new[ny,nx] = 1

#         burn = new
#         minutes += STEP_MIN
#         runs.append((minutes, burn.copy()))

#     arrival = np.full((rows, cols), np.nan)
#     for i, (m, b) in enumerate(runs):
#         arrival[(b==2) & np.isnan(arrival)] = i+1

#     # ---------- Plot ----------
#     xmin, xmax = transform.c, transform.c + transform.a * cols
#     ymin, ymax = transform.f + transform.e * rows, transform.f
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.imshow(fuel, cmap='gray_r', extent=[xmin, xmax, ymin, ymax], origin='upper')
#     im = ax.imshow(arrival, cmap=get_cmap('plasma', len(runs)),
#                    extent=[xmin, xmax, ymin, ymax], origin='upper',
#                    vmin=1, vmax=len(runs), alpha=0.75)
#     ax.set_title("Fire Arrival Time (minutes)")
#     ax.axis('off')
#     cbar = fig.colorbar(im, ax=ax, ticks=[1, len(runs)])
#     cbar.ax.set_yticklabels([f"{runs[0][0]} min", f"{runs[-1][0]} min"])
#     st.pyplot(fig)





#!/usr/bin/env python
# -------------------------------------------------
# SpreadNet Fire‑Spread Streamlit App
# -------------------------------------------------
import streamlit as st
import tensorflow as tf, numpy as np, random, math, rasterio, tempfile, requests, os
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from collections import defaultdict

# ---------- constants ----------
DEFAULT_SLOPE_URL = "https://raw.githubusercontent.com/ShayneGeo/FSIM/main/LC20_SlpD_220_SMALL2.tif"
DEFAULT_FUEL_URL  = "https://raw.githubusercontent.com/ShayneGeo/FSIM/main/LC22_F13_230_SMALL2.tif"
VALID_FUELS = [1,2,3,4,5,6,7,8,9,10,11,12,13,98,93]
FUEL_EMB = defaultdict(lambda:[0,0,0], {1:[1,0,0],2:[1,0,0],3:[1,0,0],
                                        4:[0,1,0],5:[0,1,0],6:[0,1,0],
                                        7:[0,0,1],8:[0,0,1],9:[0,0,1],
                                        10:[.5,.5,0],11:[0,.5,.5],
                                        12:[.5,0,.5],13:[.7,.3,0]})
NEIGH = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

# ---------- UI ----------
st.title("🔥 SpreadNet Fire‑Spread Simulation")
slope_file = st.file_uploader("Upload slope raster (.tif) or use default", type="tif")
fuel_file  = st.file_uploader("Upload fuel raster (.tif) or use default",  type="tif")
MOIST_GLOBAL = st.slider("Global Moisture (%)", 0.0, 40.0, 1.0)
WIND_SPEED   = st.slider("Wind Speed (m/s)", 0.0, 30.0, 10.0)
WIND_DIR_DEG = st.slider("Wind Direction (° from N)", 0, 359, 250)
STEP_MIN     = st.slider("Time Step (min)", 1, 60, 10)
MAX_SIM_MIN  = st.slider("Max Simulation Time (min)", 10, 3000, 2240)
run_btn      = st.button("Run Simulation")

# ---------- helpers ----------
def build_spreadnet():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1,  activation='sigmoid')
    ])

def download(url):
    r = requests.get(url, stream=True); r.raise_for_status()
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    for chunk in r.iter_content(1024*1024): f.write(chunk)
    f.close(); return f.name

def load_raster(path, mask_zero=False):
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
        data = np.nan_to_num(arr.filled(0))
        if mask_zero: data[data==0] = np.nan
        return data, src.transform, src.shape

def wind_align_deg(a,b): return math.cos(math.radians(a-b))
def direction_deg(y,x,ny,nx): return math.degrees(math.atan2(nx-x, ny-y))%360

def predict(net, feats): return net(np.asarray(feats,'float32'), training=False).numpy().ravel()

# ---------- main ----------
if run_btn:
    slope_path = slope_file if slope_file else download(DEFAULT_SLOPE_URL)
    fuel_path  = fuel_file  if fuel_file  else download(DEFAULT_FUEL_URL)

    slope, transform, (rows, cols) = load_raster(slope_path)
    fuel , _        , _            = load_raster(fuel_path)

    if slope.max()>90: slope = np.degrees(np.arctan(slope/100))
    slope = np.clip(slope,0,60)

    CELL, DIAG = transform.a, transform.a*math.sqrt(2)

    net = build_spreadnet()
    net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    net.fit(np.random.rand(100,8).astype('float32'),
            np.random.randint(0,2,100).astype('float32'),
            epochs=1, batch_size=32, verbose=0)

    burn = np.zeros((rows,cols), np.int8)
    burn[rows//2, cols//2] = 1
    minutes, runs = 0, []

    while burn.any() and minutes<MAX_SIM_MIN:
        new = burn.copy(); feats=[]; cells=[]
        for y,x in zip(*np.where(burn==1)):
            new[y,x]=2
            for dy,dx in NEIGH:
                ny,nx=y+dy,x+dx
                if not(0<=ny<rows and 0<=nx<cols): continue
                if burn[ny,nx]!=0 or fuel[ny,nx] in [93,98,99]: continue
                feat = FUEL_EMB[int(fuel[ny,nx])] + [
                    slope[ny,nx]/60, MOIST_GLOBAL/40, WIND_SPEED/30,
                    wind_align_deg(WIND_DIR_DEG, direction_deg(y,x,ny,nx)),
                    (DIAG if dy*dx else CELL)/DIAG]
                feats.append(feat); cells.append((ny,nx))
        if feats:
            for (ny,nx),p in zip(cells,predict(net,feats)):
                if random.random()<p: new[ny,nx]=1
        burn=new; minutes+=STEP_MIN; runs.append((minutes,burn.copy()))

    arrival=np.full((rows,cols),np.nan)
    for i,(m,b) in enumerate(runs): arrival[(b==2)&np.isnan(arrival)]=i+1

    xmin,xmax=transform.c,transform.c+transform.a*cols
    ymin,ymax=transform.f+transform.e*rows,transform.f
    fig,ax=plt.subplots(figsize=(10,10))
    ax.imshow(fuel,cmap='gray_r',extent=[xmin,xmax,ymin,ymax],origin='upper')
    im=ax.imshow(arrival,cmap=get_cmap('plasma',len(runs)),
                 extent=[xmin,xmax,ymin,ymax],origin='upper',
                 vmin=1,vmax=len(runs),alpha=.75)
    ax.set_title("Fire arrival time (min)"); ax.axis('off')
    cbar=fig.colorbar(im,ax=ax,ticks=[1,len(runs)])
    cbar.ax.set_yticklabels([f"{runs[0][0]} min",f"{runs[-1][0]} min"])
    st.pyplot(fig)

    for p in [slope_path,fuel_path]:
        if not (slope_file and p==slope_file) and not (fuel_file and p==fuel_file):
            os.remove(p)





# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import random
# import math
# import rasterio
# import matplotlib.pyplot as plt
# from matplotlib.cm import get_cmap
# from collections import defaultdict


# import tempfile
# import requests

# def download(url):
#     r = requests.get(url, stream=True)
#     r.raise_for_status()
#     f = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
#     for chunk in r.iter_content(1024 * 1024):
#         f.write(chunk)
#     f.close()
#     return f.name



# # Streamlit app layout
# st.title("Fire Spread Simulation")
# st.write("""
# ### 🔥 How SpreadNet Works

# SpreadNet replaces hand-crafted equations with a trained neural network to decide if fire spreads to a neighboring cell.

# At each time step, for each unburned neighbor, the model takes in:

# - 3 fuel type embedding values
# - Slope (normalized)
# - Moisture (normalized)
# - Wind speed (normalized)
# - Wind alignment: cos(θ) between wind and spread direction
# - Distance: straight or diagonal

# ---

# Instead of computing:

#     P(spread) = 1 - exp(-ROS × Δt / d)

# SpreadNet **learns** spread probability directly from data by asking:

# > “Given these inputs, should the fire spread here?”

# ---

# ### 🌬️ Wind in the Model

# Wind is modeled with:
# - **Wind speed** (0–1 scale)
# - **Wind alignment** (cosine of the angle between wind direction and spread direction)

# This allows the model to:
# - Favor fire spread in tailwind directions
# - Suppress spread under headwind conditions
# - Learn subtle interactions (e.g., wind affects grass differently than timber)

# ---

# ### Why Use a Neural Network?

# - Captures nonlinear relationships
# - No need to manually tune multipliers
# - However Less transparent than physics-based models

# Both models simulate fire on a grid, but:
# - The **ROS model** uses fixed rules.
# - **SpreadNet** learns its rules from examples.

# """)


# st.markdown("Adjust the parameters below to simulate fire spread using a trained neural network and cellular automaton.")

# # User inputs for tuneable constants
# MOIST_GLOBAL = st.slider("Global Moisture (%)", 0.0, 40.0, 1.0, step=0.1)
# WIND_SPEED = st.slider("Wind Speed (m/s)", 0.0, 30.0, 10.0, step=0.1)
# WIND_DIR_DEG = st.slider("Wind Direction (°)", 0, 360, 250, step=1)
# STEP_MIN = st.slider("Simulation Time Step (min)", 1, 60, 10, step=1)
# MAX_SIM_MIN = st.slider("Max Simulation Time (min)", 60, 5000, 2240, step=10)

# # File upload for raster inputs
# st.markdown("Upload the slope and fuel raster files:")
# #slope_file = st.file_uploader("Slope Raster (TIFF)", type=["tif", "tiff"])
# #fuel_file = st.file_uploader("Fuel Raster (TIFF)", type=["tif", "tiff"])


# # Default GitHub URLs for rasters
# DEFAULT_SLOPE_URL = "https://raw.githubusercontent.com/ShayneGeo/FSIM/main/LC20_SlpD_220_SMALL2.tif"
# DEFAULT_FUEL_URL  = "https://raw.githubusercontent.com/ShayneGeo/FSIM/main/LC22_F13_230_SMALL2.tif"

# slope_url = st.sidebar.text_input("Slope raster URL", DEFAULT_SLOPE_URL)
# fuel_url  = st.sidebar.text_input("Fuel model raster URL", DEFAULT_FUEL_URL)


# # Hyper-parameters for training
# N_SAMPLES = 60_000
# EPOCHS = 5
# BATCH_SIZE = 2048
# DUMMY_SEED = 42

# # Fuel embedding dictionary
# FUEL_EMB = defaultdict(lambda: [0, 0, 0], {
#     1: [1, 0, 0], 2: [1, 0, 0], 3: [1, 0, 0],
#     4: [0, 1, 0], 5: [0, 1, 0], 6: [0, 1, 0],
#     7: [0, 0, 1], 8: [0, 0, 1], 9: [0, 0, 1],
#     10: [0.5, 0.5, 0], 11: [0, 0.5, 0.5],
#     12: [0.5, 0, 0.5], 13: [0.7, 0.3, 0]
# })
# VALID_FUELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 98, 93]
# NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# def build_spreadnet():
#     return tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(8,)),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dense(16, activation='relu'),
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])

# def make_sample(label):
#     fuel_code = random.choice(VALID_FUELS)
#     fuel_emb = FUEL_EMB[fuel_code]
#     if label == 1:
#         slope = random.uniform(30, 60)
#         moist = random.uniform(0, 8)
#         wind = random.uniform(8, 30)
#         align = random.uniform(0.5, 1.0)
#     else:
#         slope = random.uniform(0, 15)
#         moist = random.uniform(25, 40)
#         wind = random.uniform(0, 10)
#         align = random.uniform(-1.0, -0.3)
#     dist = random.choice([0, 1])
#     x = fuel_emb + [slope/60, moist/40, wind/30, align, dist]
#     return x, label

# def generate_balanced_samples(n=N_SAMPLES, seed=DUMMY_SEED):
#     random.seed(seed); np.random.seed(seed)
#     half = n // 2
#     data = [make_sample(1) for _ in range(half)] + [make_sample(0) for _ in range(half)]
#     random.shuffle(data)
#     X, Y = zip(*data)
#     return np.array(X, 'float32'), np.array(Y, 'float32')

# def load_raster(file, mask_zero=False):
#     with rasterio.open(file) as src:
#         arr = src.read(1, masked=True)
#         data = np.nan_to_num(arr.filled(0))
#         if mask_zero:
#             data[data == 0] = np.nan
#         return data, src.transform, src.shape

# def wind_align_deg(dir_from, dir_to):
#     return math.cos(math.radians(dir_from - dir_to))

# def direction_deg(y, x, ny, nx):
#     return math.degrees(math.atan2(nx - x, ny - y)) % 360

# def predict_prob_batch(net, batch_feats):
#     return net(np.asarray(batch_feats, 'float32'), training=False).numpy().ravel()

# # Train the model
# if st.button("Train Model and Run Simulation"):
#     if slope_file is not None and fuel_file is not None:
#         with st.spinner("Training SpreadNet..."):
#             X, Y = generate_balanced_samples()
#             net = build_spreadnet()
#             net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#             net.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
#             st.success("SpreadNet trained successfully!")

#         with st.spinner("Running fire spread simulation..."):
#             # Load rasters
#             slope, transform, (rows, cols) = load_raster(slope_file)
#             fuel, _, _ = load_raster(fuel_file)

#             # Handle percent slope
#             if slope.max() > 90:
#                 slope = np.degrees(np.arctan(slope / 100))
#             slope = np.clip(slope, 0, 60)

#             CELL = transform.a
#             DIAG = CELL * math.sqrt(2)

#             # Run cellular automaton
#             burn = np.zeros((rows, cols), np.int8)
#             burn[rows // 2, cols // 2] = 1
#             minutes = 0
#             runs = []

#             while burn.any() and minutes < MAX_SIM_MIN:
#                 new = burn.copy()
#                 feats = []
#                 cells = []

#                 for y, x in zip(*np.where(burn == 1)):
#                     new[y, x] = 2
#                     for dy, dx in NEIGH:
#                         ny, nx = y + dy, x + dx
#                         if not (0 <= ny < rows and 0 <= nx < cols):
#                             continue
#                         if burn[ny, nx] != 0:
#                             continue
#                         if fuel[ny, nx] in [93, 98, 99]:
#                             continue

#                         emb = FUEL_EMB[int(fuel[ny, nx])]
#                         feat = emb + [
#                             slope[ny, nx] / 60,
#                             MOIST_GLOBAL / 40,
#                             WIND_SPEED / 30,
#                             wind_align_deg(WIND_DIR_DEG, direction_deg(y, x, ny, nx)),
#                             (DIAG if dy * dx else CELL) / DIAG
#                         ]
#                         feats.append(feat)
#                         cells.append((ny, nx))

#                 if feats:
#                     probs = predict_prob_batch(net, feats)
#                     for (ny, nx), p in zip(cells, probs):
#                         if random.random() < p:
#                             new[ny, nx] = 1

#                 burn = new
#                 minutes += STEP_MIN
#                 runs.append((minutes, burn.copy()))

#             # Build arrival-time map
#             arrival = np.full((rows, cols), np.nan)
#             for i, (m, b) in enumerate(runs):
#                 arrival[(b == 2) & np.isnan(arrival)] = i + 1

#             # Plot results
#             fig, ax = plt.subplots(figsize=(10, 10))
#             xmin, xmax = transform.c, transform.c + transform.a * cols
#             ymin, ymax = transform.f + transform.e * rows, transform.f
#             ax.imshow(fuel, cmap='gray_r', extent=[xmin, xmax, ymin, ymax], origin='upper')
#             im = ax.imshow(arrival, cmap=get_cmap('plasma', len(runs)),
#                            extent=[xmin, xmax, ymin, ymax], origin='upper',
#                            vmin=1, vmax=len(runs), alpha=0.75)
#             ax.set_title("Fire Arrival Time (min)")
#             ax.axis('off')
#             cbar = fig.colorbar(im, ax=ax, ticks=[1, len(runs)])
#             cbar.ax.set_yticklabels([f"{runs[0][0]} min", f"{runs[-1][0]} min"])
            
#             st.pyplot(fig)
#             st.success("Simulation complete!")
#     else:
#         st.error("Please upload both slope and fuel raster files.")

# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import random
# import math
# import rasterio
# import matplotlib.pyplot as plt
# from matplotlib.cm import get_cmap
# from collections import defaultdict
# import tempfile
# import requests
# import os

# # ------------ constants ------------ #
# DEFAULT_SLOPE_URL = "https://raw.githubusercontent.com/ShayneGeo/FSIM/main/LC20_SlpD_220_SMALL2.tif"
# DEFAULT_FUEL_URL  = "https://raw.githubusercontent.com/ShayneGeo/FSIM/main/LC22_F13_230_SMALL2.tif"
# VALID_FUELS = [1,2,3,4,5,6,7,8,9,10,11,12,13,98,93]
# FUEL_EMB = defaultdict(lambda:[0,0,0], {
#     1:[1,0,0],2:[1,0,0],3:[1,0,0],
#     4:[0,1,0],5:[0,1,0],6:[0,1,0],
#     7:[0,0,1],8:[0,0,1],9:[0,0,1],
#     10:[.5,.5,0],11:[0,.5,.5],
#     12:[.5,0,.5],13:[.7,.3,0]
# })
# NEIGH = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

# # ------------ helpers ------------ #
# def download(url):
#     r = requests.get(url, stream=True); r.raise_for_status()
#     f = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
#     for chunk in r.iter_content(1024*1024):
#         f.write(chunk)
#     f.close(); return f.name

# def load_raster(path):
#     with rasterio.open(path) as src:
#         arr = src.read(1, masked=True)
#         data = np.nan_to_num(arr.filled(0))
#         return data, src.transform, src.shape

# def wind_align(a,b): return math.cos(math.radians(a-b))
# def direction_deg(y,x,ny,nx): return math.degrees(math.atan2(nx-x, ny-y))%360

# def build_spreadnet():
#     return tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(8,)),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dense(16, activation='relu'),
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])

# def predict(net, feats):
#     return net(np.asarray(feats,'float32'), training=False).numpy().ravel()

# # ------------ UI ------------ #
# st.title("🔥 SpreadNet Fire-Spread Simulation")

# MOIST_GLOBAL = st.slider("Global Moisture (%)", 0.0, 40.0, 1.0, 0.1)
# WIND_SPEED   = st.slider("Wind Speed (m/s)",    0.0, 30.0, 10.0, 0.1)
# WIND_DIR_DEG = st.slider("Wind Direction (°)",      0, 359, 250, 1)
# STEP_MIN     = st.slider("Time Step (min)",         1, 60, 10, 1)
# MAX_SIM_MIN  = st.slider("Max Simulation Time (min)", 60, 5000, 2240, 10)

# if st.button("Run Simulation"):
#     with st.spinner("Downloading rasters..."):
#         slope_path = download(DEFAULT_SLOPE_URL)
#         fuel_path  = download(DEFAULT_FUEL_URL)

#     try:
#         with st.spinner("Preparing data..."):
#             slope, transform, (rows, cols) = load_raster(slope_path)
#             fuel , _        , _            = load_raster(fuel_path)

#             if slope.max() > 90:
#                 slope = np.degrees(np.arctan(slope / 100))
#             slope = np.clip(slope, 0, 60)
#             CELL, DIAG = transform.a, transform.a*math.sqrt(2)

#             net = build_spreadnet()
#             net.compile(optimizer='adam', loss='binary_crossentropy')
#             net.fit(np.random.rand(1000,8).astype('float32'),
#                     np.random.randint(0,2,1000).astype('float32'),
#                     epochs=1, batch_size=128, verbose=0)

#         with st.spinner("Running cellular-automaton simulation..."):
#             burn = np.zeros((rows,cols), np.int8)
#             burn[rows//2, cols//2] = 1
#             minutes, runs = 0, []

#             while burn.any() and minutes < MAX_SIM_MIN:
#                 new, feats, cells = burn.copy(), [], []
#                 for y,x in zip(*np.where(burn==1)):
#                     new[y,x] = 2
#                     for dy,dx in NEIGH:
#                         ny,nx = y+dy, x+dx
#                         if not(0<=ny<rows and 0<=nx<cols): continue
#                         if burn[ny,nx] != 0 or fuel[ny,nx] in [93,98,99]: continue
#                         feats.append(FUEL_EMB[int(fuel[ny,nx])] + [
#                             slope[ny,nx]/60, MOIST_GLOBAL/40, WIND_SPEED/30,
#                             wind_align(WIND_DIR_DEG, direction_deg(y,x,ny,nx)),
#                             (DIAG if dy*dx else CELL)/DIAG])
#                         cells.append((ny,nx))
#                 # if feats:
#                 #     for (ny,nx),p in zip(cells,predict(net,feats)):
#                 #         if random.random() < p: new[ny,nx] = 1
#                 if feats:
                            
#                             feats_np = np.asarray(feats, 'float32')
#                             if feats_np.shape[0] > 0:
#                                 probs = predict(net, feats_np)
#                                 for (ny, nx), p in zip(cells, probs):
#                                     if random.random() < p:
#                                         new[ny, nx] = 1

                        
#                 burn, minutes = new, minutes+STEP_MIN
#                 runs.append((minutes,burn.copy()))

#         arrival = np.full((rows,cols), np.nan)
#         for i,(m,b) in enumerate(runs):
#             arrival[(b==2) & np.isnan(arrival)] = i+1

#         xmin,xmax = transform.c, transform.c+CELL*cols
#         ymin,ymax = transform.f+transform.e*rows, transform.f
#         fig,ax = plt.subplots(figsize=(8,8))
#         ax.imshow(fuel,cmap='gray_r',extent=[xmin,xmax,ymin,ymax],origin='upper')
#         im = ax.imshow(arrival,cmap=get_cmap('plasma',len(runs)),
#                        extent=[xmin,xmax,ymin,ymax],origin='upper',
#                        vmin=1,vmax=len(runs),alpha=.75)
#         ax.set_title("Fire Arrival Time (min)")
#         ax.axis('off')
#         cbar = fig.colorbar(im,ax=ax,ticks=[1,len(runs)])
#         cbar.ax.set_yticklabels([f"{runs[0][0]} min",f"{runs[-1][0]} min"])
#         st.pyplot(fig)
#         st.success("Simulation complete!")

#     finally:
#         for p in [slope_path, fuel_path]:
#             try: os.remove(p)
#             except: pass
# I THINK THAT ONE WAS PRETTY GOOD

# TEST
# #!/usr/bin/env python
# # -------------------------------------------------
# # SpreadNet Fire-Spread Streamlit App (robust)
# # -------------------------------------------------
# import streamlit as st, tensorflow as tf, numpy as np, rasterio, requests, tempfile, os, math, random
# import matplotlib.pyplot as plt
# from matplotlib.cm import get_cmap
# from collections import defaultdict

# # -------------------------------------------------------------------
# # CONSTANTS
# # -------------------------------------------------------------------
# DEFAULT_SLOPE_URL = "https://raw.githubusercontent.com/ShayneGeo/FSIM/main/LC20_SlpD_220_SMALL2.tif"
# DEFAULT_FUEL_URL  = "https://raw.githubusercontent.com/ShayneGeo/FSIM/main/LC22_F13_230_SMALL2.tif"

# VALID_FUELS = [1,2,3,4,5,6,7,8,9,10,11,12,13,98,93]
# FUEL_EMB = defaultdict(lambda:[0,0,0], {
#     1:[1,0,0],2:[1,0,0],3:[1,0,0],
#     4:[0,1,0],5:[0,1,0],6:[0,1,0],
#     7:[0,0,1],8:[0,0,1],9:[0,0,1],
#     10:[.5,.5,0],11:[0,.5,.5],
#     12:[.5,0,.5],13:[.7,.3,0]
# })
# NEIGH = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

# # -------------------------------------------------------------------
# # MINI-UTILS
# # -------------------------------------------------------------------
# def download_tif(url: str) -> str:
#     """Download `url` to a temp *.tif file and return the path."""
#     r = requests.get(url, stream=True, timeout=60)
#     r.raise_for_status()
#     tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
#     for chunk in r.iter_content(1024 * 1024):
#         tmp.write(chunk)
#     tmp.close()
#     # quick sanity-check: can rasterio open it?
#     try:
#         with rasterio.open(tmp.name):
#             pass
#     except Exception:
#         os.remove(tmp.name)
#         raise RuntimeError(f"Downloaded file from {url} is not a readable GeoTIFF.")
#     return tmp.name

# def load_raster(path):
#     with rasterio.open(path) as src:
#         arr = src.read(1, masked=True)
#         return np.nan_to_num(arr.filled(0)), src.transform, src.shape

# def build_spreadnet():
#     return tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(8,)),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dense(16, activation='relu'),
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])

# def wind_align(a,b): return math.cos(math.radians(a - b))
# def direction_deg(y,x,ny,nx): return math.degrees(math.atan2(nx-x, ny-y))%360
# def predict(net, feats): return net(np.asarray(feats,'float32'), training=False).numpy().ravel()

# # -------------------------------------------------------------------
# # STREAMLIT UI
# # -------------------------------------------------------------------
# st.title("🔥 SpreadNet")
# st.write("""
# ### How it works

# SpreadNet is a neural-network–driven cellular-automaton model that predicts wildfire spread probabilities 
# for each neighboring cell based on fuel type, slope, moisture, wind speed, wind alignment, and distance.

# SpreadNet replaces CA equations with a trained neural network to decide if fire spreads to a neighboring cell.

# At each time step, for each unburned neighbor, the model takes in:

# - Fuel type embedding values
# - Slope (normalized)
# - Moisture (normalized)
# - Wind speed (normalized)
# - Wind alignment: cos(θ) between wind and spread direction
# - Distance: straight or diagonal

# ---

# Instead of computing:

#     P(spread) = 1 - exp(-ROS × Δt / d)


# SpreadNet **learns** the probability of fire spread directly from data by asking:

#     “Given these inputs, should the fire spread here?”

# This is done by training a neural network to approximate the spread probability:

#     P(spread) = σ(W₂ · ReLU(W₁ · x + b₁) + b₂)

# Where:
# - x is the input feature vector, including:
#     - fuel type,
#     - normalized slope,
#     - normalized moisture,
#     - normalized wind speed,
#     - wind alignment (cos(θ)),
#     - and a distance flag (0 = straight, 1 = diagonal)
# - σ is the sigmoid activation function converting logits to probability,
# - W₁, W₂ and b₁, b₂ are the model's learned weights and biases.

# Rather than using rule-based equations, SpreadNet learns a nonlinear decision surface
# from training data that maps environmental conditions to the probability of spread.

            
# ---

# ### 🌬️ Wind in the Model

# Wind is modeled with:
# - **Wind speed** (0–1 scale)
# - **Wind alignment** (cosine of the angle between wind direction and spread direction)

# This allows the model to:
# - Favor fire spread in tailwind directions
# - Suppress spread under headwind conditions
# - Learn subtle interactions (e.g., wind affects grass differently than timber)

# ---

# ### Why Use a Neural Network?

# - Captures nonlinear relationships
# - No need to manually tune multipliers
# - However Less transparent than physics-based models

# Both models simulate fire on a grid, but:
# - The **ROS model** uses fixed rules.
# - **SpreadNet** learns its rules from examples.

# """)
# MOIST_GLOBAL = st.slider("Global Moisture (%)", 0.0, 40.0, 1.0, .1)
# WIND_SPEED   = st.slider("Wind Speed (m/s)",    0.0, 30.0, 10.0, .1)
# WIND_DIR_DEG = st.slider("Wind Direction (°)",      0, 359, 250, 1)
# STEP_MIN     = st.slider("Time Step (min)",         1, 60, 10, 1)
# MAX_SIM_MIN  = st.slider("Max Simulation Time (min)", 60, 5000, 2240, 10)

# if st.button("Run Simulation"):

#     slope_path, fuel_path = None, None
#     try:
#         # ------------------- DOWNLOAD -------------------
#         with st.spinner("Downloading rasters…"):
#             slope_path = download_tif(DEFAULT_SLOPE_URL)
#             fuel_path  = download_tif(DEFAULT_FUEL_URL)

#         # ------------------- PREP DATA -------------------
#         slope, transform, (rows, cols) = load_raster(slope_path)
#         fuel , _        , _            = load_raster(fuel_path)

#         if slope.max() > 90:
#             slope = np.degrees(np.arctan(slope / 100))
#         slope = np.clip(slope, 0, 60)
#         CELL, DIAG = transform.a, transform.a * math.sqrt(2)

#         # ------------------- TRAIN DUMMY NET -------------
#         #rnd_X = np.random.rand(60000,8).astype('float32')
#         #rnd_Y = np.random.randint(0,2,60000).astype('float32')
        
#         # tf.keras.backend.clear_session()          # ← reset graph so reruns are clean

#         # net = build_spreadnet()
#         # net.compile(optimizer='adam', loss='binary_crossentropy')
#         # net.fit(rnd_X, rnd_Y, epochs=1, batch_size=2048, verbose=0)

#         # ---------- TRAIN DUMMY NET (wind-aware) ----------
        

            

#         def make_sample(label):
#             fuel_code = random.choice(VALID_FUELS)
#             fuel_emb  = FUEL_EMB[fuel_code]
        
#             if label:   # spread
#                 slope = random.uniform(30, 60)
#                 moist = random.uniform(0, 8)
#                 wind  = random.uniform(8, 30)
#                 align = random.uniform(0.5, 1.0)      # tail-wind
#             else:       # no-spread
#                 slope = random.uniform(0, 15)
#                 moist = random.uniform(25, 40)
#                 wind  = random.uniform(0, 10)
#                 align = random.uniform(-1.0, -0.3)    # head-wind
        
#             dist = random.choice([0, 1])
#             x = fuel_emb + [slope/60, moist/40, wind/30, align, dist]
#             return x, label
        
#         def generate_balanced_samples(n=60_000, seed=42):
#             random.seed(seed); np.random.seed(seed)
#             half = n // 2
#             data = [make_sample(1) for _ in range(half)] + \
#                    [make_sample(0) for _ in range(half)]
#             random.shuffle(data)
#             X, Y = zip(*data)
#             return np.array(X, 'float32'), np.array(Y, 'float32')
        
#         # clear previous graph each rerun
#         tf.keras.backend.clear_session()
        
#         net = build_spreadnet()
#         net.compile(optimizer='adam', loss='binary_crossentropy')
        
#         X_train, y_train = generate_balanced_samples()
#         net.fit(X_train, y_train, epochs=5, batch_size=2048, verbose=0)


#         # ------------------- SIMULATION ------------------
#         burn = np.zeros((rows,cols), np.int8)
#         burn[rows//2, cols//2] = 1
#         minutes, runs = 0, []

#         while burn.any() and minutes < MAX_SIM_MIN:
#             new, feats, cells = burn.copy(), [], []
#             for y,x in zip(*np.where(burn==1)):
#                 new[y,x] = 2
#                 for dy,dx in NEIGH:
#                     ny,nx = y+dy, x+dx
#                     if not(0<=ny<rows and 0<=nx<cols): continue
#                     if burn[ny,nx] != 0 or fuel[ny,nx] in [93,98,99]: continue
#                     feats.append(FUEL_EMB[int(fuel[ny,nx])] + [
#                         slope[ny,nx]/60, MOIST_GLOBAL/40, WIND_SPEED/30,
#                         wind_align(WIND_DIR_DEG, direction_deg(y,x,ny,nx)),
#                         (DIAG if dy*dx else CELL)/DIAG])
#                     cells.append((ny,nx))
#             if feats:
                
                        
#                 probs = predict(net, feats)
#                 for (ny,nx),p in zip(cells, probs):
#                     if random.random() < p: new[ny,nx] = 1
#             burn, minutes = new, minutes + STEP_MIN
#             runs.append((minutes, burn.copy()))

#         # ------------------- ARRIVAL MAP -----------------
#         arrival = np.full((rows,cols), np.nan)
#         for i,(_,b) in enumerate(runs):
#             arrival[(b==2) & np.isnan(arrival)] = i + 1

#         # ------------------- PLOT ------------------------
#         xmin,xmax = transform.c, transform.c + CELL * cols
#         ymin,ymax = transform.f + transform.e * rows, transform.f
#         fig, ax = plt.subplots(figsize=(8,8))
#         ax.imshow(fuel, cmap='gray_r',
#                   extent=[xmin,xmax,ymin,ymax], origin='upper')
#         im = ax.imshow(arrival, cmap=get_cmap('plasma', len(runs)),
#                        extent=[xmin,xmax,ymin,ymax], origin='upper',
#                        vmin=1, vmax=len(runs), alpha=.75)
#         ax.set_title("Fire Arrival Time (min)")
#         ax.axis('off')
#         cbar = fig.colorbar(im, ax=ax, ticks=[1, len(runs)])
#         cbar.ax.set_yticklabels([f"{runs[0][0]} min", f"{runs[-1][0]} min"])
#         st.pyplot(fig)
#         st.success("Simulation complete!")

#     finally:
#         # always clean up downloaded temp files
#         for p in (slope_path, fuel_path):
#             if p and os.path.exists(p):
#                 try: os.remove(p)
#                 except: pass














# # SLOW
# #!/usr/bin/env python
# # -------------------------------------------------
# # SpreadNet Fire-Spread Streamlit App (robust)
# # -------------------------------------------------
# import streamlit as st, tensorflow as tf, numpy as np, rasterio, requests, tempfile, os, math, random
# import matplotlib.pyplot as plt
# from matplotlib.cm import get_cmap
# from collections import defaultdict

# # -------------------------------------------------------------------
# # CONSTANTS
# # -------------------------------------------------------------------
# DEFAULT_SLOPE_URL = "https://raw.githubusercontent.com/ShayneGeo/FSIM/main/LC20_SlpD_220_SMALL2.tif"
# DEFAULT_FUEL_URL  = "https://raw.githubusercontent.com/ShayneGeo/FSIM/main/LC22_F13_230_SMALL2.tif"

# VALID_FUELS = [1,2,3,4,5,6,7,8,9,10,11,12,13,98,93]
# FUEL_EMB = defaultdict(lambda:[0,0,0], {
#     1:[1,0,0],2:[1,0,0],3:[1,0,0],
#     4:[0,1,0],5:[0,1,0],6:[0,1,0],
#     7:[0,0,1],8:[0,0,1],9:[0,0,1],
#     10:[.5,.5,0],11:[0,.5,.5],
#     12:[.5,0,.5],13:[.7,.3,0]
# })
# NEIGH = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

# # -------------------------------------------------------------------
# # MINI-UTILS
# # -------------------------------------------------------------------
# def download_tif(url: str) -> str:
#     """Download `url` to a temp *.tif file and return the path."""
#     r = requests.get(url, stream=True, timeout=60)
#     r.raise_for_status()
#     tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
#     for chunk in r.iter_content(1024 * 1024):
#         tmp.write(chunk)
#     tmp.close()
#     # quick sanity-check: can rasterio open it?
#     try:
#         with rasterio.open(tmp.name):
#             pass
#     except Exception:
#         os.remove(tmp.name)
#         raise RuntimeError(f"Downloaded file from {url} is not a readable GeoTIFF.")
#     return tmp.name

# def load_raster(path):
#     with rasterio.open(path) as src:
#         arr = src.read(1, masked=True)
#         return np.nan_to_num(arr.filled(0)), src.transform, src.shape

# def build_spreadnet():
#     return tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(8,)),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dense(16, activation='relu'),
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])

# def wind_align(a,b): return math.cos(math.radians(a - b))
# def direction_deg(y,x,ny,nx): return math.degrees(math.atan2(nx-x, ny-y))%360
# def predict(net, feats): return net(np.asarray(feats,'float32'), training=False).numpy().ravel()

# # -------------------------------------------------------------------
# # STREAMLIT UI
# # -------------------------------------------------------------------
# st.title("🔥 SpreadNet")
# st.write("""
# ### How it works

# SpreadNet is a neural-network–driven cellular-automaton model that predicts wildfire spread probabilities 
# for each neighboring cell based on fuel type, slope, moisture, wind speed, wind alignment, and distance.

# SpreadNet replaces CA equations with a trained neural network to decide if fire spreads to a neighboring cell.

# At each time step, for each unburned neighbor, the model takes in:

# - Fuel type embedding values
# - Slope (normalized)
# - Moisture (normalized)
# - Wind speed (normalized)
# - Wind alignment: cos(θ) between wind and spread direction
# - Distance: straight or diagonal

# ---

# Instead of computing:

#     P(spread) = 1 - exp(-ROS × Δt / d)


# SpreadNet **learns** the probability of fire spread directly from data by asking:

#     “Given these inputs, should the fire spread here?”

# This is done by training a neural network to approximate the spread probability:

#     P(spread) = σ(W₂ · ReLU(W₁ · x + b₁) + b₂)

# Where:
# - x is the input feature vector, including:
#     - fuel type,
#     - normalized slope,
#     - normalized moisture,
#     - normalized wind speed,
#     - wind alignment (cos(θ)),
#     - and a distance flag (0 = straight, 1 = diagonal)
# - σ is the sigmoid activation function converting logits to probability,
# - W₁, W₂ and b₁, b₂ are the model's learned weights and biases.

# Rather than using rule-based equations, SpreadNet learns a nonlinear decision surface
# from training data that maps environmental conditions to the probability of spread.

            
# ---

# ### 🌬️ Wind in the Model

# Wind is modeled with:
# - **Wind speed** (0–1 scale)
# - **Wind alignment** (cosine of the angle between wind direction and spread direction)

# This allows the model to:
# - Favor fire spread in tailwind directions
# - Suppress spread under headwind conditions
# - Learn subtle interactions (e.g., wind affects grass differently than timber)

# ---

# ### Why Use a Neural Network?

# - Captures nonlinear relationships
# - No need to manually tune multipliers
# - However Less transparent than physics-based models

# Both models simulate fire on a grid, but:
# - The **ROS model** uses fixed rules.
# - **SpreadNet** learns its rules from examples.

# """)
# MOIST_GLOBAL = st.slider("Global Moisture (%)", 0.0, 40.0, 1.0, .1)
# WIND_SPEED   = st.slider("Wind Speed (m/s)",    0.0, 30.0, 10.0, .1)
# WIND_DIR_DEG = st.slider("Wind Direction (°)",      0, 359, 250, 1)
# STEP_MIN     = st.slider("Time Step (min)",         1, 60, 10, 1)
# MAX_SIM_MIN  = st.slider("Max Simulation Time (min)", 60, 5000, 2240, 10)

# if st.button("Run Simulation"):

#     slope_path, fuel_path = None, None
#     try:
#         # ------------------- DOWNLOAD -------------------
#         with st.spinner("Downloading rasters…"):
#             slope_path = download_tif(DEFAULT_SLOPE_URL)
#             fuel_path  = download_tif(DEFAULT_FUEL_URL)

#         # ------------------- PREP DATA -------------------
#         slope, transform, (rows, cols) = load_raster(slope_path)
#         fuel , _        , _            = load_raster(fuel_path)

#         if slope.max() > 90:
#             slope = np.degrees(np.arctan(slope / 100))
#         slope = np.clip(slope, 0, 60)
#         CELL, DIAG = transform.a, transform.a * math.sqrt(2)

#         # ------------------- TRAIN DUMMY NET -------------
#         #rnd_X = np.random.rand(60000,8).astype('float32')
#         #rnd_Y = np.random.randint(0,2,60000).astype('float32')
        
#         # tf.keras.backend.clear_session()          # ← reset graph so reruns are clean

#         # net = build_spreadnet()
#         # net.compile(optimizer='adam', loss='binary_crossentropy')
#         # net.fit(rnd_X, rnd_Y, epochs=1, batch_size=2048, verbose=0)

#         # ---------- TRAIN DUMMY NET (wind-aware) ----------
        

            

#         def make_sample(label):
#             fuel_code = random.choice(VALID_FUELS)
#             fuel_emb  = FUEL_EMB[fuel_code]
        
#             if label:   # spread
#                 slope = random.uniform(30, 60)
#                 moist = random.uniform(0, 8)
#                 wind  = random.uniform(8, 30)
#                 align = random.uniform(0.5, 1.0)      # tail-wind
#             else:       # no-spread
#                 slope = random.uniform(0, 15)
#                 moist = random.uniform(25, 40)
#                 wind  = random.uniform(0, 10)
#                 align = random.uniform(-1.0, -0.3)    # head-wind
        
#             dist = random.choice([0, 1])
#             x = fuel_emb + [slope/60, moist/40, wind/30, align, dist]
#             return x, label
        
#         def generate_balanced_samples(n=60_000, seed=42):
#             random.seed(seed); np.random.seed(seed)
#             half = n // 2
#             data = [make_sample(1) for _ in range(half)] + \
#                    [make_sample(0) for _ in range(half)]
#             random.shuffle(data)
#             X, Y = zip(*data)
#             return np.array(X, 'float32'), np.array(Y, 'float32')
        
#         # clear previous graph each rerun
#         tf.keras.backend.clear_session()
        
#         net = build_spreadnet()
#         net.compile(optimizer='adam', loss='binary_crossentropy')
        
#         X_train, y_train = generate_balanced_samples()
#         net.fit(X_train, y_train, epochs=5, batch_size=2048, verbose=0)


#         # ------------------- CONSTANTS for ROS -------------
#         CELL_M  = 30.0                  # ← your request
#         # crude baseline ROS for NFDRS fuel models 1-13 (m/min)
#         ROS_FUEL = defaultdict(lambda:5, {1:6,2:7,3:9,4:5,5:4,6:3,7:1.5,
#                                           8:1.2,9:1.8,10:3,11:2,12:1.2,13:1})
#         WIND_C  = 0.23                  # m/min gain per m/s wind
#         SLOPE_C = 0.40                  # m/min gain per tan(slope)
        
#         # ------------------- SIMULATION ------------------
#         burn     = np.zeros((rows, cols), np.int8)
#         burn[rows//2, cols//2] = 1      # ignition
#         minutes, runs = 0, []
        
#         # pre-compute pixel centres for quick distance tests
#         yy, xx = np.indices(burn.shape)
        
#         while burn.any() and minutes < MAX_SIM_MIN:
#             new   = burn.copy()
#             ign_y, ign_x = np.where(burn==1)        # active flaming pixels
#             tgt_y, tgt_x, feats = [], [], []
        
#             # ---- loop over *each* flaming pixel -----------------
#             for y,x in zip(ign_y, ign_x):
#                 fuelcode = int(fuel[y,x])
#                 if fuelcode in (93,98,99):          # water / barren
#                     continue

#             # ---------------- Rate-of-Spread -----------------
#             ros = (ROS_FUEL[fuelcode] +
#                    WIND_C  * WIND_SPEED +
#                    SLOPE_C * np.tan(np.radians(slope[y,x])))
#             ros = np.clip(ros, 0.01, 200)       # m/min
#             reach_pix = ros * STEP_MIN / CELL_M # how many pixels?
    
#             # disk mask: distance from (y,x)
#             mask = ( (yy - y)**2 + (xx - x)**2 ) <= reach_pix**2
#             cand_y, cand_x = np.where(mask & (burn==0))
    
#             # ---- build NN feature vectors for all candidates
#             for ny,nx in zip(cand_y, cand_x):
#                 if fuel[ny,nx] in (93,98,99):     # skip non-burnables
#                     continue
#                 tgt_y.append(ny);  tgt_x.append(nx)
#                 feats.append( FUEL_EMB[int(fuel[ny,nx])] + [
#                     slope[ny,nx]/60,
#                     MOIST_GLOBAL/40,
#                     WIND_SPEED/30,
#                     wind_align(WIND_DIR_DEG,
#                                direction_deg(y,x,ny,nx)),
#                     int((y!=ny) and (x!=nx))      # 1 if diagonal
#                 ])

#         # ========== ignite targets probabilistically ==========
#         if feats:
#             probs = predict(net, feats)
#             for (ny,nx), p in zip(zip(tgt_y, tgt_x), probs):
#                 if random.random() < p:
#                     new[ny,nx] = 1               # becomes flaming
    
#         # update fire states
#         new[burn==1] = 2                         # flaming → burned
#         burn, minutes = new, minutes + STEP_MIN
#         runs.append((minutes, burn.copy()))


#         # ------------------- ARRIVAL MAP -----------------
#         arrival = np.full((rows,cols), np.nan)
#         for i,(_,b) in enumerate(runs):
#             arrival[(b==2) & np.isnan(arrival)] = i + 1

#         # ------------------- PLOT ------------------------
#         xmin,xmax = transform.c, transform.c + CELL * cols
#         ymin,ymax = transform.f + transform.e * rows, transform.f
#         fig, ax = plt.subplots(figsize=(8,8))
#         ax.imshow(fuel, cmap='gray_r',
#                   extent=[xmin,xmax,ymin,ymax], origin='upper')
#         im = ax.imshow(arrival, cmap=get_cmap('plasma', len(runs)),
#                        extent=[xmin,xmax,ymin,ymax], origin='upper',
#                        vmin=1, vmax=len(runs), alpha=.75)
#         ax.set_title("Fire Arrival Time (min)")
#         ax.axis('off')
#         cbar = fig.colorbar(im, ax=ax, ticks=[1, len(runs)])
#         cbar.ax.set_yticklabels([f"{runs[0][0]} min", f"{runs[-1][0]} min"])
#         st.pyplot(fig)
#         st.success("Simulation complete!")

#     finally:
#         # always clean up downloaded temp files
#         for p in (slope_path, fuel_path):
#             if p and os.path.exists(p):
#                 try: os.remove(p)
#                 except: pass


# THIS WORKE DUB TIS A BIT OK
# # requirements: streamlit, numpy, rasterio, scipy, tensorflow, matplotlib
# import streamlit as st, numpy as np, rasterio, tensorflow as tf
# from scipy.ndimage import binary_dilation
# import matplotlib.pyplot as plt, math, tempfile, requests, os, random
# from matplotlib.cm import get_cmap
# from collections import defaultdict

# # ─── constants ───────────────────────────────────────────────────────
# SLOPE_URL = "https://raw.githubusercontent.com/ShayneGeo/FSIM/main/LC20_SlpD_220_SMALL2.tif"
# FUEL_URL  = "https://raw.githubusercontent.com/ShayneGeo/FSIM/main/LC22_F13_230_SMALL2.tif"
# CELL_M    = 30.0                           # cell size (m)

# VALID = [1,2,3,4,5,6,7,8,9,10,11,12,13]
# EMB = defaultdict(lambda:[0,0,0], {
#     1:[1,0,0],2:[1,0,0],3:[1,0,0],
#     4:[0,1,0],5:[0,1,0],6:[0,1,0],
#     7:[0,0,1],8:[0,0,1],9:[0,0,1],
#     10:[.5,.5,0],11:[0,.5,.5],
#     12:[.5,0,.5],13:[.7,.3,0]})
# NONBURN = (93,98,99)
# ROS_FUEL = defaultdict(lambda:5., {1:6,2:7,3:9,4:5,5:4,6:3,7:1.5,
#                                    8:1.2,9:1.8,10:3,11:2,12:1.2,13:1})
# WIND_C, SLOPE_C = 0.23, 0.40

# # ─── utility helpers ────────────────────────────────────────────────
# def tif(url):
#     r = requests.get(url,stream=True); r.raise_for_status()
#     f = tempfile.NamedTemporaryFile(delete=False,suffix='.tif')
#     for c in r.iter_content(1024*1024): f.write(c)
#     f.close(); return f.name

# def load(path):
#     with rasterio.open(path) as s:
#         arr = s.read(1).astype('float32')
#         return arr, s.transform, s.shape

# def build_net():
#     return tf.keras.Sequential([
#         tf.keras.layers.Input((8,)),
#         tf.keras.layers.Dense(32,'relu'),
#         tf.keras.layers.Dense(16,'relu'),
#         tf.keras.layers.Dense(1,'sigmoid')
#     ])

# # ─── Streamlit UI ───────────────────────────────────────────────────
# st.title("🔥 SpreadNet – fast front-propagation")

# moist = st.slider("Fuel moisture (%)",0.,40.,1.,.1)
# wind  = st.slider("Wind speed (m s⁻¹)",0.,30.,10.,.1)
# wdir  = st.slider("Wind dir (°-from-N)",0,359,250,1)
# dt    = st.slider("Δt (min)",1,60,10,1)
# t_max = st.slider("Max time (min)",60,3000,480,10)

# if st.button("Run"):
#     slope_path = fuel_path = None
#     try:
#         slope_path, fuel_path = tif(SLOPE_URL), tif(FUEL_URL)
#         slope, tr, (rows,cols) = load(slope_path)
#         fuel , _ , _            = load(fuel_path)

#         slope = np.degrees(np.arctan(slope/100)) if slope.max()>90 else slope
#         slope = np.clip(slope,0,60)

#         # ── initialise / cache NN once ───────────────────────────
#         if 'net' not in st.session_state:
#             net = build_net()
#             net.compile(optimizer='adam', loss='binary_crossentropy')
#             net.fit(np.random.rand(2048,8), np.random.randint(0,2,2048),
#                     epochs=1,batch_size=256,verbose=0)
#             st.session_state['net'] = net
#         net = st.session_state['net']

#         # ── structuring element for dilation (single global) ─────
#         ros_max = max(ROS_FUEL.values()) + WIND_C*wind + SLOPE_C
#         rad_px  = int(np.ceil((ros_max*dt)/CELL_M))
#         yy,xs = np.ogrid[-rad_px:rad_px+1,-rad_px:rad_px+1]
#         disk   = (yy**2 + xs**2) <= rad_px**2

#         burn = np.zeros((rows,cols),np.int8)
#         burn[rows//2, cols//2] = 1
#         mins, hist = 0, []

#         y_all,x_all = np.indices((rows,cols))  # for vector maths

#         while burn.any() and mins < t_max:
#             # front = flaming cells dilated by disk
#             front = binary_dilation(burn==1, structure=disk)
#             front &= (burn==0)

#             fy, fx = np.where(front)
#             if fy.size:
#                 # build features for ALL front pixels in vectorised form
#                 emb   = np.asarray([EMB[int(c)] for c in fuel[fy,fx]])
#                 diag  = ((fy%rows)!=0)&((fx%cols)!=0)  # quick diag flag
#                 feats = np.column_stack([
#                     emb,
#                     slope[fy,fx]/60,
#                     np.full(fy.shape, moist/40,'float32'),
#                     np.full(fy.shape, wind/30 ,'float32'),
#                     np.cos(np.deg2rad(wdir -
#                         np.degrees(np.arctan2(fx-cols//2, fy-rows//2)))),
#                     diag.astype('float32')
#                 ]).astype('float32')

#                 p = net(feats,training=False).numpy().ravel()
#                 ignite = np.random.random(p.size) < p
#                 ny, nx = fy[ignite], fx[ignite]
#                 burn[ny,nx] = 1

#             burn[burn==1] = 2          # flaming → burned
#             mins += dt
#             hist.append((mins,burn.copy()))

#         # ── arrival map ───────────────────────────────────────────
#         arr = np.full((rows,cols),np.nan)
#         for i,(m,b) in enumerate(hist,1):
#             arr[(b==2)&np.isnan(arr)] = i

#         xmin,xmax = tr.c, tr.c+tr.a*cols
#         ymin,ymax = tr.f+tr.e*rows, tr.f
#         fig,ax=plt.subplots(figsize=(8,8))
#         ax.imshow(fuel,cmap='gray_r',extent=[xmin,xmax,ymin,ymax],origin='upper')
#         im=ax.imshow(arr,cmap=get_cmap('plasma',len(hist)),
#                      extent=[xmin,xmax,ymin,ymax],origin='upper',
#                      vmin=1,vmax=len(hist),alpha=.75)
#         ax.axis('off')
#         cb=fig.colorbar(im,ax=ax,ticks=[1,len(hist)])
#         cb.ax.set_yticklabels([f"{hist[0][0]} min",f"{hist[-1][0]} min"])
#         st.pyplot(fig)
#         st.success(f"Done – {len(hist)} steps ({mins} min simulated)")

#     finally:
#         for p in (slope_path,fuel_path):
#             if p and os.path.isfile(p): os.remove(p)



# # requirements: streamlit, numpy, rasterio, scipy, tensorflow, matplotlib
# import streamlit as st
# import numpy as np
# import rasterio
# import tensorflow as tf
# from scipy.ndimage import binary_dilation
# import matplotlib.pyplot as plt
# import math, tempfile, requests, os, random
# from matplotlib.cm import get_cmap
# from collections import defaultdict

# # ─── constants ───────────────────────────────────────────────────────
# SLOPE_URL = "https://raw.githubusercontent.com/ShayneGeo/FSIM/main/LC20_SlpD_220_SMALL2.tif"
# FUEL_URL  = "https://raw.githubusercontent.com/ShayneGeo/FSIM/main/LC22_F13_230_SMALL2.tif"
# CELL_M    = 30.0  # pixel size in meters

# VALID    = [1,2,3,4,5,6,7,8,9,10,11,12,13]
# EMB      = defaultdict(lambda:[0,0,0], {
#     1:[1,0,0],2:[1,0,0],3:[1,0,0],
#     4:[0,1,0],5:[0,1,0],6:[0,1,0],
#     7:[0,0,1],8:[0,0,1],9:[0,0,1],
#     10:[.5,.5,0],11:[0,.5,.5],
#     12:[.5,0,.5],13:[.7,.3,0]})
# NONBURN = (93,98,99)
# ROS_FUEL = defaultdict(lambda:5.0, {
#     1:6,2:7,3:9,4:5,5:4,6:3,7:1.5,
#     8:1.2,9:1.8,10:3,11:2,12:1.2,13:1})
# WIND_C, SLOPE_C = 0.23, 0.40  # m/min per (m/s wind), m/min per tan(slope)

# # ─── helpers ─────────────────────────────────────────────────────────
# def tif(url):
#     r = requests.get(url, stream=True)
#     r.raise_for_status()
#     f = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
#     for chunk in r.iter_content(1024*1024):
#         f.write(chunk)
#     f.close()
#     return f.name

# def load(path):
#     with rasterio.open(path) as src:
#         arr = src.read(1).astype("float32")
#         return arr, src.transform, src.shape

# def build_net():
#     m = tf.keras.Sequential([
#         tf.keras.layers.Input((8,)),
#         tf.keras.layers.Dense(32, activation="relu"),
#         tf.keras.layers.Dense(16, activation="relu"),
#         tf.keras.layers.Dense(1,  activation="sigmoid"),
#     ])
#     m.compile(optimizer="adam", loss="binary_crossentropy")
#     return m

# # ─── UI ───────────────────────────────────────────────────────────────
# st.title("🔥 SpreadNet – Wind-Oriented Front Propagation")

# moist = st.slider("Fuel moisture (%)",  0.0, 40.0, 1.0, 0.1)
# wind  = st.slider("Wind speed (m/s)",   0.0, 30.0, 10.0, 0.1)
# wdir  = st.slider("Wind dir (° from N)", 0, 359, 250, 1)
# dt    = st.slider("Δt (min)",            1, 60, 10, 1)
# tmax  = st.slider("Max time (min)",    60, 3000, 480, 10)

# if st.button("Run Simulation"):
#     slope_path = fuel_path = None
#     try:
#         # ─── load rasters ───────────────────────────────────────
#         slope_path, fuel_path = tif(SLOPE_URL), tif(FUEL_URL)
#         slope, tr, (rows,cols) = load(slope_path)
#         fuel,  _,   _          = load(fuel_path)

#         # convert percent-slope → degrees if needed
#         if slope.max() > 90:
#             slope = np.degrees(np.arctan(slope/100.0))
#         slope = np.clip(slope, 0, 60)

#         # ─── init or reuse neural net ───────────────────────────
#         if "net" not in st.session_state:
#             net = build_net()
#             # dummy train to init weights
#             X0 = np.random.rand(2048,8).astype("float32")
#             y0 = np.random.randint(0,2,2048).astype("float32")
#             net.fit(X0, y0, epochs=1, batch_size=256, verbose=0)
#             st.session_state.net = net
#         net = st.session_state.net

#         # ─── build wind-oriented ellipse SE ─────────────────────
#         # maximum possible ROS (m/min) for any fuel+slope
#         max_slope = np.tan(np.deg2rad(slope.max()))
#         ros_dw = max(ROS_FUEL.values()) + WIND_C*wind + SLOPE_C*max_slope
#         ros_cw = max(ROS_FUEL.values()) +             SLOPE_C*max_slope
#         # radii in pixels
#         a = (ros_dw * dt) / CELL_M
#         b = (ros_cw * dt) / CELL_M
#         theta = np.deg2rad(wdir)
#         rad = int(np.ceil(max(a,b)))
#         yy, xx = np.ogrid[-rad:rad+1, -rad:rad+1]
#         # rotate coords
#         xr =  xx*np.cos(theta) + yy*np.sin(theta)
#         yr = -xx*np.sin(theta) + yy*np.cos(theta)
#         ellipse = (xr/a)**2 + (yr/b)**2 <= 1

#         # ─── simulate ────────────────────────────────────────────
#         burn = np.zeros((rows,cols), np.int8)
#         burn[rows//2, cols//2] = 1  # ignition at center
#         elapsed, history = 0, []

#         while burn.any() and elapsed < tmax:
#             # grow the front by the oriented ellipse
#             front = binary_dilation(burn==1, structure=ellipse) & (burn==0)
#             fy, fx = np.where(front)

#             if fy.size:
#                 # build features vectorized for all front pixels
#                 emb = np.asarray([ EMB[int(c)] for c in fuel[fy,fx] ], "float32")
#                 # slope, moisture, wind, alignment, diagonal flag
#                 diag = ((fy % rows)!=0) & ((fx % cols)!=0)
#                 align = np.cos(
#                     np.deg2rad(wdir - 
#                         np.degrees(np.arctan2(fx-cols//2, fy-rows//2)))
#                 )
#                 feats = np.column_stack([
#                     emb,
#                     slope[fy,fx]/60.0,
#                     np.full(fy.shape, moist/40.0, "float32"),
#                     np.full(fy.shape, wind /30.0,  "float32"),
#                     align.astype("float32"),
#                     diag.astype("float32"),
#                 ]).astype("float32")

#                 p = net(feats, training=False).numpy().ravel()
#                 ignite = (np.random.rand(p.size) < p)
#                 ny, nx = fy[ignite], fx[ignite]
#                 burn[ny, nx] = 1

#             # move flaming → burned, advance time
#             burn[burn==1] = 2
#             elapsed += dt
#             history.append((elapsed, burn.copy()))

#         # ─── arrival time map ────────────────────────────────────
#         arrival = np.full((rows,cols), np.nan)
#         for i,(t, bmap) in enumerate(history, start=1):
#             arrival[(bmap==2)&np.isnan(arrival)] = i

#         # ─── plot ────────────────────────────────────────────────
#         xmin,xmax = tr.c, tr.c + tr.a*cols
#         ymin,ymax = tr.f + tr.e*rows, tr.f
#         fig,ax = plt.subplots(figsize=(8,8))
#         ax.imshow(fuel, cmap="gray_r",
#                   extent=[xmin,xmax,ymin,ymax], origin="upper")
#         im = ax.imshow(arrival, cmap=get_cmap("plasma", len(history)),
#                        extent=[xmin,xmax,ymin,ymax], origin="upper",
#                        vmin=1, vmax=len(history), alpha=0.75)
#         ax.axis("off")
#         cb = fig.colorbar(im, ax=ax, ticks=[1, len(history)])
#         cb.ax.set_yticklabels([f"{history[0][0]} min", f"{history[-1][0]} min"])
#         st.pyplot(fig)
#         st.success(f"Done – {len(history)} steps (~{elapsed} min)")

#     finally:
#         for p in (slope_path, fuel_path):
#             if p and os.path.isfile(p):
#                 os.remove(p)


