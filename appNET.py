# =====================================================
# 1.  TRAIN A SIMPLE SpreadNet WITH DUMMY DATA
# =====================================================
import tensorflow as tf, numpy as np, random, os, warnings
warnings.filterwarnings("ignore")

def build_spreadnet():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(9,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1,  activation="sigmoid")
    ])
    return model

def make_sample():
    fuel = random.choices([0,1,2],[0.4,0.4,0.2])[0]
    fuel_onehot = [int(fuel==i) for i in range(3)]
    slope = random.uniform(0,60)
    moist = random.uniform(0,40)
    wind  = random.uniform(0,30)
    align = random.uniform(0,1)
    dist  = random.choice([0,1])
    x = fuel_onehot + [slope/60, moist/40, wind/30, align, dist]
    fuel_mult = [0.3,0.6,0.9][fuel]
    prob = fuel_mult*(1-moist/40)*(0.3+0.7*align)*(0.2+0.8*wind/30)
    y = 1 if prob>0.3 else 0
    return x,y

X,Y=zip(*(make_sample() for _ in range(60000)))
X,Y=np.array(X,dtype="float32"),np.array(Y,dtype="float32")
model=build_spreadnet()
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.fit(X,Y,epochs=5,batch_size=4096,verbose=0)
model.save_weights("spread_nn_tf.h5")
print("✅  Dummy SpreadNet trained & saved\n")

# =====================================================
# 2.  STREAMLIT FIRE‑SPREAD SIM USING THAT MODEL
# =====================================================
import streamlit as st, rasterio, requests, tempfile
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap, BoundaryNorm
from collections import defaultdict
import math

st.title("Neural‑Network Fire Spread Simulation (TensorFlow)")

DEFAULT_SLOPE="https://raw.githubusercontent.com/ShayneGeo/FSIM/main/LC20_SlpD_220_SMALL2.tif"
DEFAULT_FUEL ="https://raw.githubusercontent.com/ShayneGeo/FSIM/main/LC22_F13_230_SMALL2.tif"
slp_url=st.sidebar.text_input("Slope raster URL",DEFAULT_SLOPE)
ful_url=st.sidebar.text_input("Fuel raster URL",DEFAULT_FUEL)
W_SPD=st.sidebar.slider("Wind speed (m/s)",0.0,30.0,10.0)
W_DIR=st.sidebar.slider("Wind dir (°)",0,360,210)
MOIST=st.sidebar.slider("Dead fuel moisture %",0.0,40.0,2.0)
STEP =st.sidebar.slider("Time‑step (min)",1,60,10)
MAXT =st.sidebar.slider("Max sim time (min)",60,14400,2400)
run  =st.sidebar.button("Run")

@st.cache_data
def load_r(url):
    if url.startswith("http"):
        r=requests.get(url);r.raise_for_status()
        tmp=tempfile.NamedTemporaryFile(suffix=".tif",delete=False)
        tmp.write(r.content);tmp.flush();pth=tmp.name
    else: pth=url
    with rasterio.open(pth) as src:
        data=src.read(1,masked=True) if 'LC20' in url else src.read(1)
        return np.nan_to_num(data.filled(0)),src.transform,src.shape
slope,tr,shape=load_r(slp_url);fuel,tr2,_=load_r(ful_url)
rows,cols=shape;CELL=tr.a;diag=CELL*math.sqrt(2)
if slope.max()>90:slope=np.degrees(np.arctan(slope/100))
slope=np.clip(slope,0,60)

FUEL_EMB=defaultdict(lambda:[0,0,0],{
    1:[1,0,0],2:[1,0,0],3:[1,0,0],
    4:[0,1,0],5:[0,1,0],6:[0,1,0],
    7:[0,0,1],8:[0,0,1],9:[0,0,1],
    10:[.5,.5,0],11:[0,.5,.5],12:[.5,0,.5],13:[.7,.3,0]})
nbrs=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

tf_model=build_spreadnet();tf_model.load_weights("spread_nn_tf.h5")

def p_spread(y,x,ny,nx):
    emb=FUEL_EMB[int(fuel[ny,nx])]
    feat=emb+[slope[ny,nx]/60,MOIST/40,W_SPD/30,
              (math.cos(math.radians(W_DIR-w_angle(y,x,ny,nx)))+1)/2,
              (diag if (ny-y)*(nx-x) else CELL)/diag]
    return float(tf_model(np.array(feat,dtype="float32").reshape(1,-1))[0,0])

def w_angle(y,x,ny,nx):
    dy,dx=ny-y,nx-x
    return math.degrees(math.atan2(dx,dy))%360

if run:
    burn=np.zeros((rows,cols),np.int8)
    burn[rows//2,cols//2]=1
    minutes=0;runs=[]
    while burn.any()==1 and minutes<MAXT:
        new=burn.copy()
        for y,x in zip(*np.where(burn==1)):
            new[y,x]=2
            for dy,dx in nbrs:
                ny,nx=y+dy,x+dx
                if 0<=ny<rows and 0<=nx<cols and burn[ny,nx]==0:
                    if np.random.rand()<p_spread(y,x,ny,nx):new[ny,nx]=1
        burn=new;minutes+=STEP
        if minutes%STEP==0:runs.append((minutes,burn.copy()))
    arr=np.full((rows,cols),np.nan)
    for i,(m,b) in enumerate(runs):
        mask=(b==2);arr[mask & np.isnan(arr)]=i+1
    xmin,xmax=tr.c,tr.c+tr.a*cols
    ymin,ymax=tr.f+tr.e*rows,tr.f
    fig,ax=plt.subplots(figsize=(8,8))
    ax.imshow(fuel,cmap="gray_r",origin="upper",extent=[xmin,xmax,ymin,ymax])
    im=ax.imshow(arr,cmap=get_cmap("plasma",len(runs)),vmin=1,vmax=len(runs),
                 alpha=.75,origin="upper",extent=[xmin,xmax,ymin,ymax])
    ax.axis("off"); c=fig.colorbar(im,ax=ax,ticks=[1,len(runs)])
    c.ax.set_yticklabels([f"{runs[0][0]} min",f"{runs[-1][0]} min"])
    c.set_label("Fire arrival time"); st.pyplot(fig)
else:
    st.info("Adjust settings and click ▶ Run Simulation")
