import streamlit as st
import numpy as np, random, math, tensorflow as tf, rasterio
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from collections import defaultdict
from io import BytesIO

# Hyperparameters
N_SAMPLES = 60000
EPOCHS = 5
BATCH_SIZE = 2048
DUMMY_SEED = 42

st.set_page_config(layout="wide")
st.title("🔥 SpreadNet: Fire Spread Simulation with Neural Net + Cellular Automaton")

# ----------------------------------------
# Model definition
# ----------------------------------------
def build_spreadnet():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

FUEL_EMB = defaultdict(lambda:[0,0,0], {
    1:[1,0,0],2:[1,0,0],3:[1,0,0],
    4:[0,1,0],5:[0,1,0],6:[0,1,0],
    7:[0,0,1],8:[0,0,1],9:[0,0,1],
    10:[.5,.5,0],11:[0,.5,.5],
    12:[.5,0,.5],13:[.7,.3,0]
})

VALID_FUELS = list(FUEL_EMB.keys()) + [93, 98, 99]

def make_sample(label):
    fuel_code = random.choice(VALID_FUELS)
    fuel_emb = FUEL_EMB[fuel_code]
    if label == 1:
        slope = random.uniform(30, 60)
        moist = random.uniform(0, 8)
        wind = random.uniform(8, 30)
        align = random.uniform(0.5, 1.0)
    else:
        slope = random.uniform(0, 15)
        moist = random.uniform(25, 40)
        wind = random.uniform(0, 10)
        align = random.uniform(-1.0, -0.3)
    dist = random.choice([0, 1])
    x = fuel_emb + [slope/60, moist/40, wind/30, align, dist]
    return x, label

def generate_balanced_samples(n=N_SAMPLES, seed=DUMMY_SEED):
    random.seed(seed)
    np.random.seed(seed)
    half = n // 2
    data = [make_sample(1) for _ in range(half)] + [make_sample(0) for _ in range(half)]
    random.shuffle(data)
    X, Y = zip(*data)
    return np.array(X, 'float32'), np.array(Y, 'float32')

# ----------------------------------------
# Training
# ----------------------------------------
st.header("1️⃣ Train SpreadNet")
if st.button("Train Neural Network"):
    X, Y = generate_balanced_samples()
    net = build_spreadnet()
    net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    net.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    st.success("✅ Model trained!")

    # Save in session state
    st.session_state.net = net
    st.session_state.trained = True
else:
    net = st.session_state.get("net", None)

# ----------------------------------------
# Raster upload
# ----------------------------------------
st.header("2️⃣ Upload Raster Files")

col1, col2 = st.columns(2)
with col1:
    slope_file = st.file_uploader("Upload slope raster (GeoTIFF)", type=["tif"], key="slope")
with col2:
    fuel_file = st.file_uploader("Upload fuel raster (GeoTIFF)", type=["tif"], key="fuel")

def load_raster(file):
    with rasterio.open(file) as src:
        arr = src.read(1, masked=True)
        data = np.nan_to_num(arr.filled(0))
        return data, src.transform, src.shape

# ----------------------------------------
# Simulation
# ----------------------------------------
st.header("3️⃣ Simulate Fire Spread")

MOIST_GLOBAL = st.slider("Moisture (%)", 0.0, 40.0, 1.0)
WIND_SPEED = st.slider("Wind speed (m/s)", 0.0, 30.0, 10.0)
WIND_DIR_DEG = st.slider("Wind direction (deg)", 0, 360, 250)
STEP_MIN = 10
MAX_SIM_MIN = 2240

if st.button("Run Simulation") and slope_file and fuel_file and net:
    slope, transform, (rows, cols) = load_raster(slope_file)
    fuel, _, _ = load_raster(fuel_file)

    if slope.max() > 90:
        slope = np.degrees(np.arctan(slope / 100))
    slope = np.clip(slope, 0, 60)

    CELL = transform.a
    DIAG = CELL * math.sqrt(2)
    NEIGH = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def wind_align_deg(dir_from, dir_to):
        return math.cos(math.radians(dir_from - dir_to))

    def direction_deg(y,x,ny,nx):
        return math.degrees(math.atan2(nx-x, ny-y)) % 360

    def predict_prob_batch(batch_feats):
        return net(np.asarray(batch_feats, 'float32'), training=False).numpy().ravel()

    burn = np.zeros((rows, cols), np.int8)
    burn[rows//2, cols//2] = 1
    minutes = 0
    runs = []

    while burn.any() and minutes < MAX_SIM_MIN:
        new = burn.copy()
        feats = []
        cells = []

        for y, x in zip(*np.where(burn == 1)):
            new[y, x] = 2
            for dy, dx in NEIGH:
                ny, nx = y+dy, x+dx
                if not (0 <= ny < rows and 0 <= nx < cols): continue
                if burn[ny, nx] != 0: continue
                if fuel[ny, nx] in [93, 98, 99]: continue

                emb = FUEL_EMB[int(fuel[ny, nx])]
                feat = emb + [
                    slope[ny, nx]/60,
                    MOIST_GLOBAL/40,
                    WIND_SPEED/30,
                    wind_align_deg(WIND_DIR_DEG, direction_deg(y, x, ny, nx)),
                    (DIAG if dy*dx else CELL)/DIAG
                ]
                feats.append(feat)
                cells.append((ny, nx))

        if feats:
            probs = predict_prob_batch(feats)
            for (ny, nx), p in zip(cells, probs):
                if random.random() < p:
                    new[ny, nx] = 1

        burn = new
        minutes += STEP_MIN
        runs.append((minutes, burn.copy()))

    # Arrival time map
    arrival = np.full((rows, cols), np.nan)
    for i, (m, b) in enumerate(runs):
        arrival[(b==2) & np.isnan(arrival)] = i + 1

    # Plot
    xmin, xmax = transform.c, transform.c + transform.a*cols
    ymin, ymax = transform.f + transform.e*rows, transform.f
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(fuel, cmap='gray_r', extent=[xmin,xmax,ymin,ymax], origin='upper')
    im = ax.imshow(arrival, cmap=get_cmap('plasma', len(runs)),
                   extent=[xmin,xmax,ymin,ymax], origin='upper',
                   vmin=1, vmax=len(runs), alpha=0.75)
    ax.set_title("Fire Arrival Time (min)")
    ax.axis('off')
    cbar = fig.colorbar(im, ax=ax, ticks=[1, len(runs)])
    cbar.ax.set_yticklabels([f"{runs[0][0]} min", f"{runs[-1][0]} min"])
    st.pyplot(fig)

elif not net:
    st.warning("❌ Please train the model first.")
elif not slope_file or not fuel_file:
    st.warning("❌ Please upload both slope and fuel raster files.")
