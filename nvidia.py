# ===============================================
# 1. Import Libraries
# ===============================================
# Menjalankan streamlit :
# py -m streamlit run nvidia.py    (untuk menjalankan streamlit)

# Menyimpan output terminal:
# py nvidia.py |Tee-Object -FilePath ".\output_terminal.txt"

# Data Analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Dashboard
import streamlit as st
# Yahoo Finance
import yfinance as yf   
# File Management  
import os

# Machine Learning 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,r2_score)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Ignore Warning
import warnings
warnings.filterwarnings("ignore")

# CREATE FOLDERS
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("output/figures", exist_ok=True)
os.makedirs("output/predictions", exist_ok=True)

# ==================================================
# 2. DATA COLLECTION
# ==================================================

data_nvidia = yf.download("NVDA",start="2023-08-01",end="2026-08-01",interval="1d",auto_adjust=False,progress=False)

# Rapikan kolom yfinance
if isinstance(data_nvidia.columns, pd.MultiIndex):
    data_nvidia.columns = data_nvidia.columns.get_level_values(0)
# Date dari index menjadi kolom
data_nvidia = data_nvidia.reset_index()
data_nvidia.to_csv("data/raw/nvidia_stock_3years.csv",index=False)

print("Data NVIDIA berhasil diambil dari Yahoo Finance.")
print("Periode: 1 Agustus 2023 - 31 Juli 2026")
print("Ukuran dataset:", data_nvidia.shape)

# ==================================================
# 3. DATA CLEANING
# ==================================================

# Salin raw dataset 
df = data_nvidia.copy()        
# Bersihkan nama kolom                         
df.columns = (df.columns.astype(str).str.strip()) 
# Konversi Date:difilter berdasarkan periode, dan digunakan pada visualisasi time series.      
df["Date"] = pd.to_datetime(df["Date"])    
# Konversi kolom numerik            
kolom_numerik = ["Open","High","Low","Close","Adj Close","Volume"] 
for kolom in kolom_numerik:
    if kolom in df.columns:
        df[kolom] = pd.to_numeric(df[kolom])
# Hapus missing values        
df = df.dropna(subset=["Date","Open","High","Low","Close","Volume"]) 
# Hapus duplikat
df = df.drop_duplicates()  
# Urutkan berdasarkan tanggal
df = df.sort_values("Date")  
# Reset index
df.reset_index(drop=True,inplace=True)  
# Lokasi Simpan clean dataset
clean_path = "data/processed/nvidia_stock_3years_clean.csv"  
df.to_csv(clean_path,index=False)  

print("\nClean dataset berhasil disimpan.")
print(f"Lokasi: {clean_path}")

# Cleaning Validation
print("\nNama Kolom:")
print(df.columns.tolist())
print("\nMissing Values Setelah Cleaning:")
print(df.isnull().sum())
print("\nDuplicate Rows Setelah Cleaning:")
print(df.duplicated().sum())
print("\nDataset Information:")
df.info()

# tabel missing value
missing = pd.DataFrame({
    "Missing Values": df.isnull().sum(),
    "Percentage (%)": round(df.isnull().mean()*100,2)})
print("\nMissing Value Summary:")
print(missing)

# =================================================
#  4. Exploratory Data Analysis
# =================================================

print("="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# Ukuran dataset
print("\nUkuran Dataset:")
print(f"\nJumlah Baris : {df.shape[0]}")
print(f"Jumlah Kolom : {df.shape[1]}")

# Periode data
print("\nPeriode Data")
print("Tanggal Awal :", df["Date"].min())
print("Tanggal Akhir :", df["Date"].max())

# Statistik harga
print("\nHarga Saham")
print(f"Harga Tertinggi : ${df['High'].max():.2f}")
print(f"Harga Terendah  : ${df['Low'].min():.2f}")

# Closing Price
print("\nClosing Price")
print(f"Rata-rata : ${df['Close'].mean():.2f}")
print(f"Median    : ${df['Close'].median():.2f}")

# Trading Volume
print("\nTrading Volume")
print(f"Total Volume    : {df['Volume'].sum():,.0f}")
print(f"Rata-rata Volume: {df['Volume'].mean():,.0f}")

# Statistik Deskriptif
print("\nStatistik Deskriptif:")
print(df.describe())

# Data awal dan akhir
print("\n5 Data Pertama:")
print(df.head())
print("\n5 Data Terakhir:")
print(df.tail())

# summary
summary = pd.DataFrame({
    "Metric":["Total Trading Days","Highest Close","Lowest Close","Average Close","Highest Volume"],
    "Value":[len(df),df["Close"].max(),df["Close"].min(),round(df["Close"].mean(),2),df["Volume"].max()]})
print("\nEDA Summary:")
print(summary)

# ===============================================
# 5. VISUALIZATION 1 MONTH LATEST
# ===============================================

# Filter data 1 bulan terakhir
tanggal_akhir = df["Date"].max()
tanggal_awal = tanggal_akhir - pd.DateOffset(months=1)

df_plot = df[df["Date"] >= tanggal_awal].copy()

# Menghitung informasi utama
harga_awal = df_plot["Close"].head(1).values[0]
harga_akhir = df_plot["Close"].tail(1).values[0]
harga_tertinggi = df_plot["Close"].max()
harga_terendah = df_plot["Close"].min()
perubahan_persen = ((harga_akhir - harga_awal)/ harga_awal) * 100

# Volume dalam satuan juta
volume_juta = df_plot["Volume"] / 1_000_000
rata_rata_volume = volume_juta.mean()

# Hijau jika harga naik, merah jika harga turun
warna_volume = np.where(df_plot["Close"] >= df_plot["Open"],"green","red")

# MEMBUAT 2 GRAFIK (GRAFIK A.HARGA dan B.VOLUME)

fig, (ax1, ax2) = plt.subplots(2,1,
    figsize=(15, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]})

# A. GRAFIK HARGA

ax1.plot(df_plot["Date"],df_plot["Close"],
    marker="o",
    linewidth=2)

ax1.fill_between(df_plot["Date"],df_plot["Close"],harga_terendah,
    alpha=0.12)

# Angka di setiap titik harga
for tanggal, harga in zip(df_plot["Date"],df_plot["Close"]):
    ax1.text(tanggal,harga + 0.35,f"{harga:.2f}",
        ha="center",
        fontsize=7)

# Kotak ringkasan
ringkasan = (
    f"Start: ${harga_awal:.2f}\n"
    f"Latest: ${harga_akhir:.2f}\n"
    f"High: ${harga_tertinggi:.2f}\n"
    f"Low: ${harga_terendah:.2f}\n"
    f"Change: {perubahan_persen:+.2f}%")

ax1.text(0.02,0.95,ringkasan,
    transform=ax1.transAxes,
    va="top",
    fontsize=10,
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.90))

ax1.set_title("NVIDIA Price and Trading Volume — Last 1 Month",
    fontsize=17,
    fontweight="bold",
    loc="left")

ax1.set_ylabel("Closing Price (USD)")
ax1.grid(alpha=0.20)

# B.GRAFIK VOLUME

bars = ax2.bar(df_plot["Date"],volume_juta,
    color=warna_volume,
    alpha=0.55)

# Angka vertikal di dalam bar
for bar, volume in zip(bars,volume_juta):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() / 2,
        f"{volume:.1f}M",
        ha="center",
        va="center",
        rotation=90,
        fontsize=7)

# Garis rata-rata volume
ax2.axhline(rata_rata_volume,
    linestyle="--",
    linewidth=1,
    label=f"Average: {rata_rata_volume:.2f}M")

ax2.set_ylabel("Volume (Million)")
ax2.set_xlabel("Date")
ax2.legend(loc="upper right")
ax2.grid(axis="y", alpha=0.20)


# FORMAT AND SAVE
for ax in [ax1, ax2]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("output/figures/nvidia_price_volume_1month.png",
    dpi=300,
    bbox_inches="tight")
plt.show()

# ==================================================
# 6. MACHINE LEARNING SUPERVISED & UNSUPERVISED
# =================================================

# =================================================
# SUPERVISED :
# 6.1 RIDGE REGRESSION & DECISION TREE REGRESSION
# =================================================

# 6.1.1 PREPROCESSING DAN FEATURE ENGINEERING

ml_data = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
ml_data = ml_data.sort_values("Date")                  #Data diurutkan berdasarkan tanggal.

# Membuat feature tambahan
ml_data["Close_Lag_1"] = ml_data["Close"].shift(1)      #Close_Lag_1 adalah harga penutupan satu hari sebelumnya.
ml_data["MA_5"] = ml_data["Close"].rolling(5).mean()    #MA_5 :rata-rata harga penutupan selama lima hari terakhir. (tren jangka pendek.)
ml_data["MA_20"] = ml_data["Close"].rolling(20).mean()  #MA_20 :rata-rata harga penutupan selama 20 hari terakhir.(tren jangka menengah)
ml_data["Daily_Return"] = ml_data["Close"].pct_change() #Daily_Return menghitung persentase perubahan harga penutupan dibandingkan hari sebelumnya.
# Target: harga penutupan hari berikutnya
ml_data["Next_Close"] = ml_data["Close"].shift(-1)       #Next_Close adalah harga penutupan hari berikutnya.

# 6.1.2 FEATURE SELECTION

fitur = ["Close","Close_Lag_1","MA_5","MA_20","Daily_Return","Volume"]

# Ambil data terbaru untuk prediksi masa depan
data_terbaru = ml_data.dropna(subset=fitur).tail(1)
X_terbaru = data_terbaru[fitur]
# Hapus baris yang belum mempunyai target
ml_data = ml_data.dropna(subset=fitur + ["Next_Close"]).copy()
# data model
X = ml_data[fitur]
Y = ml_data["Next_Close"]

# 6.1.3 SPLIT DATA

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20,shuffle=False)  #20% data digunakan untuk testing
tanggal_test = ml_data["Date"].tail(len(X_test))
print("=" * 50)
print("DATA SPLIT")
print("=" * 50)
print("Jumlah Data Training:", len(X_train))
print("Jumlah Data Testing :", len(X_test))

# 6.1.4  RIDGE REGRESSION

model_ridge = Pipeline([("scaler", StandardScaler()),("model", Ridge(alpha=1.0))])  
model_ridge.fit(X_train,y_train)   
# Prediction                                                
y_pred_ridge = model_ridge.predict(X_test)
# Evaluation
mae_ridge = mean_absolute_error(y_test,y_pred_ridge)                      #MAE:rata-rata selisih absolut antara harga aktual dan harga prediksi.
mape_ridge = mean_absolute_percentage_error(y_test,y_pred_ridge) * 100    #MAPE menghitung rata-rata error dalam bentuk persentase.  
rmse_ridge = np.sqrt(mean_squared_error(y_test,y_pred_ridge))             #RMSE juga mengukur error prediksi
r2_ridge = r2_score(y_test,y_pred_ridge)                                  #R² mengukur seberapa baik model menjelaskan variasi harga aktual.

print("=" * 50)
print("RIDGE REGRESSION PERFORMANCE")
print("=" * 50)
print("MAE             :", round(mae_ridge, 2))
print("MAPE            :", round(mape_ridge, 2), "%")
print("RMSE            :", round(rmse_ridge, 2))
print("R²              :", round(r2_ridge, 4))

# 6.1.5 DECISION TREE REGRESSION

# Membuat model Decision Tree
model_tree = DecisionTreeRegressor()
# Training
model_tree.fit(X_train,y_train)
# Prediksi
y_pred_tree = model_tree.predict(X_test)
# Evaluasi
mae_tree = mean_absolute_error(y_test,y_pred_tree)
mape_tree = mean_absolute_percentage_error(y_test,y_pred_tree) * 100
rmse_tree = np.sqrt(mean_squared_error(y_test,y_pred_tree))
r2_tree = r2_score(y_test,y_pred_tree)

print("=" * 50)
print("DECISION TREE PERFORMANCE")
print("=" * 50)
print("MAE      :", round(mae_tree, 2))
print("MAPE     :", round(mape_tree, 2), "%")
print("RMSE     :", round(rmse_tree, 2))
print("R²       :", round(r2_tree, 4))


# 6.1.6 BASELINE DAN MODEL COMPARISON

# Baseline:Tomorrow Close = Today Close (benchmark sederhana)
baseline_prediksi = X_test["Close"]
baseline_mae = mean_absolute_error(y_test,baseline_prediksi)

print("=" * 50)
print("MODEL COMPARISON")
print("=" * 50)
print("Ridge MAE         :",round(mae_ridge, 2))
print("Decision Tree MAE :",round(mae_tree, 2))
print("Baseline MAE      :",round(baseline_mae, 2))

# Tentukan hasil terbaik
if (mae_ridge < mae_tree and mae_ridge < baseline_mae):
    print("Kesimpulan: Ridge Regression " 
          "memiliki MAE paling kecil.")
elif (mae_tree < mae_ridge and mae_tree < baseline_mae):
    print("Kesimpulan: Decision Tree "
        "memiliki MAE paling kecil.")
else:
    print("Kesimpulan: Baseline "
        "masih memiliki MAE paling kecil.")
    

# 6.1.7 PREDICTION RESULT

hasil_prediksi = pd.DataFrame({
    "Date": tanggal_test.values,
    "Actual_Close": y_test.values,
    "Ridge_Prediction":y_pred_ridge,
    "Decision_Tree_Prediction":y_pred_tree,
    "Baseline_Close": baseline_prediksi.values})

# Error Ridge
hasil_prediksi["Ridge_Absolute_Error"] = abs(hasil_prediksi["Actual_Close"]- hasil_prediksi["Ridge_Prediction"])
# Error Decision Tree
hasil_prediksi["Tree_Absolute_Error"] = abs(hasil_prediksi["Actual_Close"]- hasil_prediksi["Decision_Tree_Prediction"])

print("\n5 Hasil Prediksi:")
print(hasil_prediksi.head())


# 6.1.8 ACTUAL VS MODEL PREDICTION

fig, ax = plt.subplots(figsize=(15, 7))

# Actual Close
ax.plot(hasil_prediksi["Date"],hasil_prediksi["Actual_Close"],
    label="Actual Close",
    color="#3A7CA5",
    linewidth=3.0)

# Ridge Prediction
ax.plot(hasil_prediksi["Date"],hasil_prediksi["Ridge_Prediction"],
    label="Ridge Prediction",
    color="#F28E2B",
    linewidth=2.4,
    linestyle="--")

# Decision Tree
ax.plot(hasil_prediksi["Date"],hasil_prediksi["Decision_Tree_Prediction"],
    label="Decision Tree Prediction",
    color="#59A14F",
    linewidth=2.0,
    linestyle="-.")

# Baseline Prediction
ax.plot(hasil_prediksi["Date"],hasil_prediksi["Baseline_Close"],
    label="Baseline: Previous Close",
    color="#7A7A7A",
    linewidth=1.8,
    linestyle=":",
    alpha=0.9)

ax.set_title("NVIDIA Closing Price Prediction Comparison",
    fontsize=19,
    fontweight="bold",
    loc="left")

ax.set_xlabel("Date",fontsize=12)
ax.set_ylabel("Closing Price (USD)",fontsize=12)
ax.legend(loc="upper right")
ax.grid(alpha=0.20)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Format tanggal
ax.xaxis.set_major_formatter( mdates.DateFormatter("%b %Y"))

# Ringkasan performa model
latest_actual = hasil_prediksi["Actual_Close"].iloc[-1]
baseline_mape = (mean_absolute_percentage_error(y_test,baseline_prediksi) * 100)
model_summary = (
    f"Latest Actual: ${latest_actual:.2f}\n"
    f"Ridge MAE: ${mae_ridge:.2f}|"
    f"MAPE: {mape_ridge:.2f}%\n"
    f"Decision Tree MAE: ${mae_tree:.2f}|"
    f"MAPE: {mape_tree:.2f}%\n"
    f"Baseline MAE: ${baseline_mae:.2f}|"
    f"MAPE: {baseline_mape:.2f}%")

# Menampilkan kotak ringkasan pada grafik

ax.text(0.015,0.97,model_summary,
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=11,
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        edgecolor="#C9C9C9",
        alpha=0.95))

plt.xticks(rotation=45)
plt.tight_layout()

file_grafik = os.path.join("output","figures","nvidia_model_comparison.png")
plt.savefig(file_grafik,dpi=300,bbox_inches="tight")
print("Grafik berhasil disimpan:", file_grafik)

plt.show()


# 6.1.9 NEXT TRADING DAY PREDICTION

harga_close_terbaru = (data_terbaru["Close"].values[0])
tanggal_terbaru = (data_terbaru["Date"].values[0])

# Ridge Prediction
prediksi_ridge_next = (model_ridge.predict(X_terbaru)[0])
perubahan_ridge = (prediksi_ridge_next- harga_close_terbaru)
persen_ridge = (perubahan_ridge/ harga_close_terbaru) * 100

# Decision Tree Prediction
prediksi_tree_next = (model_tree.predict(X_terbaru)[0])
perubahan_tree = (prediksi_tree_next- harga_close_terbaru)
persen_tree = (perubahan_tree/ harga_close_terbaru) * 100

print("=" * 50)
print("NEXT TRADING DAY PREDICTION")
print("=" * 50)

print("Tanggal terakhir:",pd.to_datetime(tanggal_terbaru).strftime("%d %B %Y"))
print("Latest Close:",f"${harga_close_terbaru:,.2f}")
print("=" * 20)
print("RIDGE REGRESSION")
print("=" * 20)
print("Predicted Close:",f"${prediksi_ridge_next:,.2f}")
print("Change:",f"${perubahan_ridge:+,.2f}")
print("Percentage:",f"{persen_ridge:+,.2f}%")
print("=" * 20)
print("DECISION TREE")
print("=" * 20)
print("Predicted Close:",f"${prediksi_tree_next:,.2f}")
print("Change:",f"${perubahan_tree:+,.2f}")
print("Percentage:",f"{persen_tree:+,.2f}%")

# 6.1.10 SAVE PREDICTION RESULT

hasil_prediksi.to_csv("output/predictions/""nvidia_model_predictions.csv",index=False)
print("\nHasil prediksi berhasil disimpan.")

# ==================================================
# MACHINE LEARNING UNSUPERVISED :
# 6.2 K-MEANS
# ==================================================

# 6.2.1 Siapkan Data
data_kmeans = df.copy()
# Membuat Daily Return
data_kmeans["Daily_Return"] = (data_kmeans["Close"].pct_change() * 100)
# Hapus data kosong
data_kmeans = data_kmeans.dropna()
# 6.2.2 Feature Selection
X = data_kmeans[["Daily_Return", "Volume"]]

# 6.2.3 Training
model_kmeans = KMeans(
    n_clusters=2,
    init="k-means++",
)
# 6.2.4 Clustering
cluster = model_kmeans.fit_predict(X)
data_kmeans["Cluster"] = cluster

# 6.2.5 Simpan Hasil
data_kmeans.to_csv("output/predictions/nvidia_kmeans_result.csv")

# 6.2.6 Evaluasi
score_kmeans = silhouette_score(X,cluster)
print("\nSilhouette Score K-Means:")
print(score_kmeans)

# 6.2.7 Cluster Summary
cluster_summary = data_kmeans.groupby("Cluster")[["Daily_Return", "Volume"]].mean()
print("\nK-Means Cluster Summary:")
print(cluster_summary)


# 6.2.8 VISUALISASI K-MEANS
plt.figure(figsize=(10, 6))

nama_cluster = {
    0: "Higher Trading Activity",
    1: "Lower Trading Activity"}


for i in range(2):
    data_cluster = data_kmeans[data_kmeans["Cluster"] == i]
    plt.scatter(
        data_cluster["Daily_Return"],
        data_cluster["Volume"],
         label=nama_cluster[i])

plt.xlabel("Daily Return (%)")
plt.ylabel("Trading Volume")
plt.title("NVIDIA Trading Day Clusters")
plt.legend()
plt.savefig("output/figures/nvidia_kmeans_clusters.png",dpi=300,bbox_inches="tight")
plt.show()


# ==================================================
# 7. STREAMLIT DASHBOARD
# ==================================================

# 7.1 KONFIGURASI HALAMAN

st.set_page_config(
    page_title="NVIDIA Stock Analytics",
    page_icon="📈",
    layout="wide")


# 7.2 CSS DASHBOARD

st.markdown(
    """
    <style>

    .block-container {padding-top: 3.5rem;padding-bottom: 2rem;max-width: 1400px;}
    [data-testid="stSidebar"] {background-color: #6F8F7B;border-right: 1px solid #D9E2DC;}
    [data-testid="stSidebar"] * {color: #FFFFFF;}
    [data-testid="stSidebar"] h1 {font-size: 28px !important;}
    [data-testid="stSidebar"] p {font-size: 16px !important;}
    [data-testid="stSidebar"] div[role="radiogroup"] label p {font-size: 17px !important;}
    [data-testid="stAppViewContainer"] {background-color: #F5F7F5;}
    div[data-testid="stRadio"] {margin-top: 0.5rem;margin-bottom: 1.5rem;}

    </style>
    """,
    unsafe_allow_html=True
)


# 7.3 LOAD CLEAN DATA

@st.cache_data
def load_clean_data():

    data = pd.read_csv("data/processed/nvidia_stock_3years_clean.csv")
    data["Date"] = pd.to_datetime(data["Date"],errors="coerce")
    kolom_numerik = ["Open","High","Low","Close","Volume"]

    for kolom in kolom_numerik:
        data[kolom] = pd.to_numeric(data[kolom],errors="coerce")
    data = data.dropna(subset=["Date","Open","High","Low","Close","Volume"])
    data = data.sort_values("Date")
    data.reset_index(drop=True,inplace=True)
    return data

data_dashboard = load_clean_data()

# 7.4 FORMAT ANGKA

def format_juta(nilai):
    return f"{nilai / 1_000_000:,.2f}M"
def format_miliar(nilai):
    return f"{nilai / 1_000_000_000:,.2f}B"


# 7.5 SIDEBAR NAVIGATION

st.sidebar.title("📊 NVIDIA Analytics")
st.sidebar.caption("Stock Analytics & Prediction")
st.sidebar.markdown("---")
st.sidebar.markdown("**NVIDIA Corporation**")
st.sidebar.caption("Ticker: NVDA · NasdaqGS")

halaman = st.sidebar.radio("Navigation",["Summary","Historical Data","Price Analysis","Trading Volume",
                                         "Statistics","Prediction", "K-Means","Business Insight"])
st.sidebar.markdown("---")
st.sidebar.caption("Daily stock data · Aug 2023 – Jul 2026")

# 7.6 FILTER PERIODE

tanggal_terakhir = data_dashboard["Date"].max()

def filter_periode(data, periode):

    tanggal_akhir = data["Date"].max()

    if periode == "1M":tanggal_awal = (tanggal_akhir- pd.DateOffset(months=1))
    elif periode == "3M":tanggal_awal = (tanggal_akhir- pd.DateOffset(months=3))
    elif periode == "6M":tanggal_awal = (tanggal_akhir- pd.DateOffset(months=6))
    elif periode == "YTD":tanggal_awal = pd.Timestamp(year=tanggal_akhir.year,month=1,day=1)
    elif periode == "1Y":tanggal_awal = (tanggal_akhir- pd.DateOffset(years=1))
    else:
        tanggal_awal = (tanggal_akhir- pd.DateOffset(years=3))
    data_filter = data[data["Date"] >= tanggal_awal].copy()
    return data_filter


# 7.7  HALAMAN STREAMLIT

# ==================================================
# 7.7.1 HALAMAN SUMMARY
# ==================================================

if halaman == "Summary":

    # FILTER PERIODE
    periode = st.radio("Periode",["1M", "3M", "6M", "YTD", "1Y", "3Y"],
        index=3,
        horizontal=True,
        label_visibility="collapsed",
        key="periode_summary")

    data_filter = filter_periode(data_dashboard,periode)

    if data_filter.empty:
        st.warning("Tidak ada data pada periode yang dipilih.")
        st.stop()

    # PERHITUNGAN SUMMARY
   
    harga_terakhir = data_filter["Close"].iloc[-1]
    harga_awal_periode = data_filter["Close"].iloc[0]
    perubahan_periode = (harga_terakhir- harga_awal_periode)
    persentase_periode = (perubahan_periode/ harga_awal_periode) * 100

    # JUDUL
    st.title("NVIDIA Corporation (NVDA)")
    st.caption("Analysis based on NVIDIA daily stock data from "
            "1 August 2023 to 31 July 2026.")

    
    # CHART HARGA DAN VOLUME
    # grafik interaktif gunakan Plotly go.Figure()dan go.Scatter() dan tampilkan dgn st.plotly_chart()
    # hovertemplate : membuat user bisa arahkan mouse ke grafik dan melihat detail
    # hovermode="x unified" : membuat informasi hover mengikuti posisi tanggal pada sumbu X.

    warna_volume = np.where(data_filter["Close"] >= data_filter["Open"],
            "#43a77b",
            "#e87970") 
        

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25])

    # Closing Price
    fig.add_trace(
        go.Scatter(x=data_filter["Date"],y=data_filter["Close"],
            mode="lines",
            name="Close",
            line=dict(width=3.0),
            customdata=np.stack(
            ( data_filter["Open"],data_filter["High"],data_filter["Low"],data_filter["Volume"]),
            axis=-1),

            hovertemplate=(
                "<b>Date:</b> %{x|%d %b %Y}<br>"
                "<b>Open:</b> $%{customdata[0]:,.2f}<br>"
                "<b>High:</b> $%{customdata[1]:,.2f}<br>"
                "<b>Low:</b> $%{customdata[2]:,.2f}<br>"
                "<b>Close:</b> $%{y:,.2f}<br>"
                "<b>Volume:</b> %{customdata[3]:,.0f}"
                "<extra></extra>")),
        row=1,
        col=1)

    fig.update_yaxes(
        title_text="Price (USD)",
        showgrid=True,
        row=1,
        col=1)

    # Trading Volume
    fig.add_trace(
        go.Bar(x=data_filter["Date"],y=data_filter["Volume"],
            marker_color=warna_volume,
            name="Volume",
            hovertemplate=(
                "Date: %{x|%d %b %Y}<br>"
                "Volume: %{y:,.0f}"
                "<extra></extra>")),
        row=2,
        col=1)
    
    fig.update_yaxes(
        title_text="Volume",
        showgrid=False,
        row=2,
        col=1)

    fig.update_layout(
        height=650,
        hovermode="x unified",
        showlegend=False)

    st.plotly_chart(fig,use_container_width=True)

    # MARKET INSIGHT
  
    if persentase_periode > 20:status_indo = "tren kenaikan yang kuat"
    elif persentase_periode > 0:status_indo = "tren positif"
    elif persentase_periode == 0:status_indo = "tren yang relatif stabil"
    else:
        status_indo = "tren penurunan"

    st.info(
        f"NVIDIA menunjukkan {status_indo} selama periode {periode}. "
        f"Harga berubah dari ${harga_awal_periode:.2f} "
        f"menjadi ${harga_terakhir:.2f}, "
        f"atau {persentase_periode:+.2f}%.")

    # MARKET SUMMARY
   
    st.markdown("---")
    st.subheader("Market Summary")

    # Data terbaru
    previous_close = data_dashboard["Close"].iloc[-2]
    open_price = data_dashboard["Open"].iloc[-1]
    latest_close = data_dashboard["Close"].iloc[-1]
    volume_terakhir = data_dashboard["Volume"].iloc[-1]
    # Ringkasan data 3 tahun
    total_trading_days = len(data_dashboard)
    average_close = data_dashboard["Close"].mean()

    highest_price = data_dashboard["High"].max()
    lowest_price = data_dashboard["Low"].min()

    average_volume_3y = data_dashboard["Volume"].mean()

    tanggal_awal = data_dashboard["Date"].min()
    tanggal_akhir = data_dashboard["Date"].max()

   
    # LATEST MARKET DATA
   
    st.subheader("Latest Market Data")
    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Previous Close",f"${previous_close:.2f}")
    m2.metric("Open",f"${open_price:.2f}")
    m3.metric("Latest Close",f"${latest_close:.2f}")
    m4.metric("Volume",format_juta(volume_terakhir))


    # 3-YEAR DATASET SUMMARY

    st.subheader("3-Year Dataset Summary")
    m5, m6, m7 = st.columns(3)

    m5.metric("Trading Days",f"{total_trading_days:,}")
    m6.metric("Average Close",f"${average_close:.2f}")
    m7.metric("Average Daily Volume",format_juta(average_volume_3y))

    m8, m9, m10 = st.columns(3)

    m8.metric("Highest Price",f"${highest_price:.2f}")
    m9.metric("Lowest Price",f"${lowest_price:.2f}")
    m10.metric("Data Period",f"{tanggal_awal:%b %Y} - {tanggal_akhir:%b %Y}")
    
# ==================================================
# 7.7.2 HALAMAN HISTORICAL DATA
# ==================================================

elif halaman == "Historical Data":

    st.header("Historical Data")
    st.write("Pilih periode untuk menampilkan data historis NVIDIA.")

    # FILTER PERIODE
    periode_historical = st.radio("Pilih Periode",["1M", "3M", "6M", "YTD", "1Y", "3Y"],
        index=5,
        horizontal=True,
        key="periode_historical")

    historical_filter = filter_periode(data_dashboard,periode_historical)

    # Urutkan dari tanggal terbaru
    historical_filter = historical_filter.sort_values("Date",ascending=False)

    # KPI
    col1, col2, col3 = st.columns(3)

    col1.metric("Selected Period",periode_historical)
    col2.metric("Trading Days",len(historical_filter))
    col3.metric("Date Range",f"{historical_filter['Date'].min():%d %b %Y} - "f"{historical_filter['Date'].max():%d %b %Y}")

    # TABEL
    st.dataframe(historical_filter,
        hide_index=True,
        use_container_width=True,
        height=450)

    # DOWNLOAD CSV
    st.download_button(label="Download Historical Data",data=historical_filter.to_csv(index=False),
        file_name=f"nvidia_{periode_historical.lower()}.csv",
        mime="text/csv")

# ==================================================
# 7.7.3 HALAMAN PRICE ANALYSIS
# ==================================================

elif halaman == "Price Analysis":

    st.header("Price Analysis")
    st.write("Pilih periode untuk melihat pergerakan harga NVIDIA.")

    # FILTER PERIODE
    periode_price = st.radio("Pilih Periode",["1M", "3M", "6M", "YTD", "1Y", "3Y"],
        index=3,
        horizontal=True,
        key="periode_price_analysis")

    price_filter = filter_periode(data_dashboard,periode_price)

    # KPI
    harga_awal_price = price_filter["Close"].iloc[0]
    harga_akhir_price = price_filter["Close"].iloc[-1]
    perubahan_price = (harga_akhir_price - harga_awal_price)
    persentase_price = (perubahan_price / harga_awal_price) * 100

    p1, p2, p3 = st.columns(3)

    p1.metric("Start Price",f"${harga_awal_price:.2f}")
    p2.metric("Latest Price",f"${harga_akhir_price:.2f}")
    p3.metric("Period Change",f"{persentase_price:+.2f}%")

    # PILIH JENIS HARGA
    kolom_harga_dipilih = st.multiselect("Pilih Jenis Harga",["Open", "High", "Low", "Close"],
        default=["Open", "High", "Low", "Close"])

    if not kolom_harga_dipilih:
        st.warning("Pilih minimal satu jenis harga.")
        st.stop()


    fig_price = go.Figure()

    for kolom in kolom_harga_dipilih:
        fig_price.add_trace(
            go.Scatter( x=price_filter["Date"],y=price_filter[kolom],
                mode="lines",
                name=kolom,
                hovertemplate=(
                    f"{kolom}<br>"
                    "Date: %{x|%d %b %Y}<br>"
                    "Price: $%{y:.2f}"
                    "<extra></extra>")))

    fig_price.update_layout(
        title=f"NVIDIA Price Analysis — {periode_price}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=520,
        hovermode="x unified")

    st.plotly_chart(fig_price,use_container_width=True)

    st.markdown("---")
    st.subheader("Candlestick Chart") #untuk menampilkan Open, High, Low, dan Close dalam satu grafik

    fig_candle = go.Figure()
    fig_candle.add_trace(
        go.Candlestick(
            x=price_filter["Date"],
            open=price_filter["Open"],
            high=price_filter["High"],
            low=price_filter["Low"],
            close=price_filter["Close"],
            name="NVDA"))
    fig_candle.update_layout(
        title=f"NVIDIA Candlestick — {periode_price}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=550,
        xaxis_rangeslider_visible=False)

    st.plotly_chart(fig_candle,use_container_width=True)

# ==================================================
# 7.7.4 HALAMAN TRADING VOLUME
# ==================================================

elif halaman == "Trading Volume":

    st.header("Trading Volume")
    st.write("Pilih periode untuk melihat volume perdagangan harian NVIDIA.")

    # FILTER PERIODE
    periode_volume = st.radio("Pilih Periode",["1M", "3M", "6M", "YTD", "1Y", "3Y"],
        index=3,
        horizontal=True,
        key="periode_trading_volume")

    volume_filter = filter_periode(data_dashboard,periode_volume)

    if volume_filter.empty:
        st.warning("Tidak ada data volume untuk periode yang dipilih.")
        st.stop()


    # PERHITUNGAN KPI
    average_volume = volume_filter["Volume"].mean()
    highest_volume = volume_filter["Volume"].max()
    tanggal_volume_tertinggi = volume_filter.loc[volume_filter["Volume"].idxmax(),"Date"]


    # KPI
    col1, col2, col3 = st.columns(3)

    col1.metric("Average Daily Volume",f"{average_volume / 1_000_000:.2f}M")
    col2.metric("Highest Volume",f"{highest_volume / 1_000_000:.2f}M")
    col3.metric("Trading Days",len(volume_filter))

    # WARNA VOLUME
    warna_volume = np.where(volume_filter["Close"] >= volume_filter["Open"],
        "#43a77b",
        "#e87970") 
   
    # GRAFIK INTERAKTIF
    fig_volume = go.Figure()
    fig_volume.add_trace(
        go.Bar(x=volume_filter["Date"],y=volume_filter["Volume"],
            marker_color=warna_volume,
            hovertemplate=(
                "Date: %{x|%d %b %Y}<br>"
                "Volume: %{y:,.0f}"
                "<extra></extra>")))

    fig_volume.update_layout(
        title=f"NVIDIA Trading Volume — {periode_volume}",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=500,
        showlegend=False)

    st.plotly_chart(fig_volume,use_container_width=True)


    # VOLUME INSIGHT
    st.info(
        f"Average daily volume: "
        f"{average_volume / 1_000_000:.2f}M shares. "
        f"Highest volume: "
        f"{highest_volume / 1_000_000:.2f}M shares "
        f"on {tanggal_volume_tertinggi:%d %B %Y}.")


# ==================================================
# 7.7.5 HALAMAN STATISTICS
# ==================================================

elif halaman == "Statistics":

    st.header("Statistics")
    st.subheader("Price Statistics — 3-Year Dataset")
    st.caption ("Statistics are calculated from NVIDIA daily stock data "
                 "covering 1 August 2023 to 31 July 2026.")
        
    # Data satu tahun terakhir
    data_52_week = data_dashboard[data_dashboard["Date"]>= tanggal_terakhir - pd.DateOffset(years=1)]

    # STATISTIC 
    # iloc berarti mengambil data berdasarkan posisi baris.

    previous_close = data_dashboard["Close"].iloc[-2]
    open_price = data_dashboard["Open"].iloc[-1]

    volume_terakhir = data_dashboard["Volume"].iloc[-1]
    average_volume = data_52_week["Volume"].mean()

    total_trading_days = len(data_dashboard)
    average_close = data_dashboard["Close"].mean()

    # KPI

    kolom1, kolom2, kolom3 = st.columns(3)

    kolom1.metric("Previous Close",f"${previous_close:.2f}")
    kolom2.metric("Open",f"${open_price:.2f}")
    kolom3.metric("Average Close (3Y)",f"${average_close:.2f}")


    kolom4, kolom5, kolom6 = st.columns(3)

    kolom4.metric("Volume",format_juta(volume_terakhir))
    kolom5.metric("Average Volume (52W)",format_juta(average_volume))
    kolom6.metric("Trading Days",f"{total_trading_days:,}")

    # PRICE STATISTICS
    st.markdown("---")
    st.subheader("Price Statistics")

    statistik_harga = pd.DataFrame({
        "Metric": ["Average","Median","Minimum","Maximum","Standard Deviation"],

        "Open": [
            data_dashboard["Open"].mean(),
            data_dashboard["Open"].median(),
            data_dashboard["Open"].min(),
            data_dashboard["Open"].max(),
            data_dashboard["Open"].std()],

        "High": [
            data_dashboard["High"].mean(),
            data_dashboard["High"].median(),
            data_dashboard["High"].min(),
            data_dashboard["High"].max(),
            data_dashboard["High"].std()],

        "Low": [
            data_dashboard["Low"].mean(),
            data_dashboard["Low"].median(),
            data_dashboard["Low"].min(),
            data_dashboard["Low"].max(),
            data_dashboard["Low"].std()],

        "Close": [
            data_dashboard["Close"].mean(),
            data_dashboard["Close"].median(),
            data_dashboard["Close"].min(),
            data_dashboard["Close"].max(),
            data_dashboard["Close"].std()]})

    kolom_harga = ["Open","High","Low","Close"]
    statistik_harga[kolom_harga] = (statistik_harga[kolom_harga].round(2))
    st.dataframe(statistik_harga,hide_index=True,use_container_width=True)

    st.markdown("---")
    st.subheader("Closing Price Distribution")

    fig_hist = go.Figure()

    fig_hist.add_trace(
        go.Histogram(
            x=data_dashboard["Close"],
            nbinsx=30,   #harga Close dibagi menjadi sekitar 30 interval/rentang harga.  
            marker_color="#76B900",
            hovertemplate=(
                "Close Price: $%{x:.2f}<br>"
                "Frequency: %{y}"
                "<extra></extra>")))

    fig_hist.update_layout(
            title="Distribution of NVIDIA Closing Price — 3 Years",
            xaxis_title="Close Price (USD)",
            yaxis_title="Frequency",
            height=450,
            showlegend=False)

    st.plotly_chart(fig_hist,use_container_width=True)

    st.subheader("Data Quality")
    jumlah_missing = int(data_dashboard.isnull().sum().sum())
    jumlah_duplikat = int(data_dashboard.duplicated().sum())
    kelengkapan_data = (1- jumlah_missing / data_dashboard.size) * 100

    kualitas1, kualitas2, kualitas3 = st.columns(3)

    kualitas1.metric("Missing Values",jumlah_missing)
    kualitas2.metric("Duplicate Rows",jumlah_duplikat)
    kualitas3.metric("Data Completeness",f"{kelengkapan_data:.2f}%")

    if jumlah_missing == 0 and jumlah_duplikat == 0:
        st.success("Dataset lengkap dan tidak memiliki data duplikat.")
    else:
        st.warning(
            "Dataset masih memiliki missing values "
            "atau data duplikat.")

# ==================================================
# 7.7.6 HALAMAN PREDICTION
# ==================================================

elif halaman == "Prediction":

    st.header("Machine Learning Prediction")

    st.write(
        "Halaman ini menampilkan perbandingan performa model "
        "dan prediksi harga penutupan NVIDIA.")

    st.info(
        "Model digunakan untuk memprediksi harga penutupan "
        "pada hari perdagangan berikutnya berdasarkan data historis.")

    
    # KPI PERFORMA MODEL
    latest_actual = hasil_prediksi["Actual_Close"].iloc[-1]
    p1, p2, p3, p4, p5 = st.columns(5)

    p1.metric("Latest Actual",f"${latest_actual:,.2f}")
    p2.metric("Ridge MAE",f"${mae_ridge:,.2f}")
    p3.metric("Ridge MAPE",f"{mape_ridge:.2f}%")
    p4.metric("Decision Tree MAE",f"${mae_tree:,.2f}")
    p5.metric("Baseline MAE",f"${baseline_mae:,.2f}")

    
    # ACTUAL VS PREDICTION

    st.subheader("Actual vs Prediction")
    st.image("output/figures/nvidia_model_comparison.png",use_container_width=True)

    
    # INTERPRETASI MODEL
    
    st.subheader("Model Interpretation")

    st.write(
        """
        Berdasarkan visualisasi perbandingan, Ridge Regression mampu
        mengikuti pola pergerakan harga penutupan aktual NVIDIA dengan
        cukup baik. Hasil prediksi Ridge terlihat dekat dengan baseline
        yang menggunakan harga penutupan sebelumnya sebagai prediksi.

        Sementara itu, Decision Tree Regression menunjukkan perubahan
        prediksi yang lebih kaku dan pada beberapa periode menghasilkan
        garis mendatar. Hal ini menunjukkan bahwa Decision Tree kurang
        mampu mengikuti perubahan harga yang berada di luar pola data
        pelatihannya.
        """
    )

    # Kesimpulan otomatis berdasarkan MAE

    if (baseline_mae <= mae_ridge and baseline_mae <= mae_tree):
        st.warning(
            "Baseline memiliki MAE paling kecil. Artinya, model "
            "machine learning belum mampu meningkatkan akurasi "
            "dibandingkan metode sederhana yang menggunakan harga "
            "penutupan sebelumnya.")
    elif mae_ridge <= mae_tree:
        st.success(
            "Ridge Regression memiliki MAE paling kecil sehingga "
            "menjadi model dengan performa terbaik pada data pengujian.")
    else:
        st.success(
            "Decision Tree Regression memiliki MAE paling kecil "
            "sehingga menjadi model dengan performa terbaik pada "
            "data pengujian.")


    # NEXT TRADING DAY PREDICTION

    st.subheader("Next Trading Day Prediction")

    n1, n2, n3 = st.columns(3)

    n1.metric(
        "Latest Close",
        f"${harga_close_terbaru:,.2f}")

    n2.metric(
        "Ridge Prediction",
        f"${prediksi_ridge_next:,.2f}",
        f"{persen_ridge:+,.2f}%")

    n3.metric(
        "Decision Tree Prediction",
        f"${prediksi_tree_next:,.2f}",
        f"{persen_tree:+,.2f}%")

    st.caption(
        f"Prediksi dibuat menggunakan data terakhir tanggal "
        f"{pd.to_datetime(tanggal_terbaru):%d %B %Y}.")

    st.warning(
        "Prediksi dibuat berdasarkan data historis dan bukan "
        "merupakan rekomendasi investasi.")

# ==================================================
# HALAMAN K-MEANS
# ==================================================

elif halaman == "K-Means":

    st.header("K-Means Clustering")

    st.write(
        "K-Means digunakan untuk mengelompokkan hari perdagangan "
        "NVIDIA berdasarkan Daily Return dan Trading Volume.")

    # KPI
    st.metric("Silhouette Score",f"{score_kmeans:.3f}")

    # CLUSTER SUMMARY
    st.subheader("Cluster Summary")
    cluster_tampil = cluster_summary.copy()
    cluster_tampil["Daily_Return"] = (cluster_tampil["Daily_Return"].map("{:.2f}%".format))
    cluster_tampil["Volume"] = (cluster_tampil["Volume"].map("{:,.0f}".format))
    cluster_tampil["Trading Activity"] = ["Lower Trading Activity","Higher Trading Activity"]

    # Tentukan cluster dengan volume tertinggi
    cluster_volume = cluster_summary["Volume"]
    cluster_tampil["Trading Activity"] = [
        "Higher Trading Activity"
        if volume == cluster_volume.max()
        else "Lower Trading Activity"
        for volume in cluster_volume]
    
    st.dataframe(cluster_tampil,use_container_width=True)

    # VISUALISASI
    st.subheader("Trading Day Clusters")
    st.image("output/figures/nvidia_kmeans_clusters.png",use_container_width=900)

    # INTERPRETASI
    st.info(
         "Cluster 0 represents higher trading activity, "
        "while Cluster 1 represents lower trading activity.")
    
# ==================================================
# 7.7.7 HALAMAN BUSINESS INSIGHT
# ==================================================

elif halaman == "Business Insight":

    st.header("Business Insight")

    st.write("Ringkasan performa harga, risiko, dan aktivitas "
            "perdagangan NVIDIA berdasarkan periode yang dipilih.")

    # FILTER PERIODE

    periode_insight = st.radio(
        "Pilih Periode Analisis",["1M","3M","6M","YTD","1Y","3Y"],
        index=3,
        horizontal=True,
        key="periode_business_insight")

    insight_data = filter_periode(data_dashboard,periode_insight)

    if insight_data.empty:
        st.warning("Tidak ada data untuk periode yang dipilih.")
        st.stop()


    # PERHITUNGAN INSIGHT
   
    harga_awal_insight = insight_data["Close"].iloc[0]
    harga_akhir_insight = insight_data["Close"].iloc[-1]
    perubahan_insight = (harga_akhir_insight- harga_awal_insight)
    persentase_insight = (perubahan_insight/ harga_awal_insight) * 100
    harga_tertinggi_insight = insight_data["Close"].max()
    harga_terendah_insight = insight_data["Close"].min()
    tanggal_tertinggi_insight = insight_data.loc[insight_data["Close"].idxmax(),"Date"]
    tanggal_terendah_insight = insight_data.loc[insight_data["Close"].idxmin(),"Date"]
    rata_rata_close_insight = insight_data["Close"].mean()
    rata_rata_volume_insight = insight_data["Volume"].mean()
    volume_tertinggi_insight = insight_data["Volume"].max()
    tanggal_volume_tertinggi = insight_data.loc[insight_data["Volume"].idxmax(),"Date"]
    insight_data["Daily Return"] = (insight_data["Close"].pct_change())
    volatilitas_harian = (insight_data["Daily Return"].std() * 100)
    volatilitas_tahunan = (insight_data["Daily Return"].std()* np.sqrt(252)* 100)


    # KATEGORI TREND

    if persentase_insight >= 20:
        status_tren = "Strong Growth"
        status_deskripsi = ("Harga menunjukkan tren kenaikan yang kuat.")
    elif persentase_insight > 0:
        status_tren = "Moderate Growth"
        status_deskripsi = ("Harga menunjukkan pertumbuhan positif.")
    elif persentase_insight == 0:
        status_tren = "Stable"
        status_deskripsi = ("Harga relatif tidak mengalami perubahan.")
    elif persentase_insight > -20:
        status_tren = "Moderate Decline"
        status_deskripsi = ("Harga mengalami penurunan moderat.")
    else:
        status_tren = "Strong Decline"
        status_deskripsi = ("Harga mengalami tekanan penurunan yang kuat.")


    # KATEGORI VOLATILITAS

    if volatilitas_tahunan < 20:kategori_risiko = "Low Volatility"
    elif volatilitas_tahunan < 40:kategori_risiko = "Moderate Volatility"
    else:
        kategori_risiko = "High Volatility"


    # KPI UTAMA

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric("Price Change",f"{persentase_insight:+,.2f}%",f"${perubahan_insight:+,.2f}")
    kpi2.metric("Trend Category",status_tren)
    kpi3.metric("Annualized Volatility",f"{volatilitas_tahunan:,.2f}%")
    kpi4.metric("Trading Days",f"{len(insight_data):,}")

    st.caption(
        f"Analysis period: "
        f"{insight_data['Date'].min():%d %B %Y} – "
        f"{insight_data['Date'].max():%d %B %Y}")

   
    # PERFORMANCE OVERVIEW
   
    st.subheader("Performance Overview")

    overview1, overview2, overview3, overview4 = st.columns(4)

    overview1.metric("Starting Price",f"${harga_awal_insight:,.2f}")
    overview2.metric("Latest Price",f"${harga_akhir_insight:,.2f}")
    overview3.metric("Highest Close",f"${harga_tertinggi_insight:,.2f}")
    overview4.metric("Lowest Close",f"${harga_terendah_insight:,.2f}")

    # TREND CHART
    # grafik interaktif gunakan Plotly go.Figure()dan go.Scatter() dan tampilkan dgn st.plotly_chart()
    # hovertemplate : membuat user bisa arahkan mouse ke grafik dan melihat detail
    # hovermode="x unified" : membuat informasi hover mengikuti posisi tanggal pada sumbu X.
    st.subheader("Price Trend and Average Price")
    fig_insight = go.Figure()
    fig_insight.add_trace(
        go.Scatter(x=insight_data["Date"],y=insight_data["Close"],
            mode="lines",
            name="Close Price",
            hovertemplate=(
                "<b>Date:</b> %{x|%d %b %Y}<br>"
                "<b>Close:</b> $%{y:,.2f}"
                "<extra></extra>")))

    fig_insight.add_hline(y=rata_rata_close_insight,
        line_dash="dash",
        annotation_text=(f"Average Close ${rata_rata_close_insight:,.2f}"),
        annotation_position="top left")

    fig_insight.update_layout(height=460,
        title=(f"NVIDIA Closing Price — {periode_insight}"),
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=20,r=20,t=60,b=20))

    fig_insight.update_xaxes(showgrid=False)
    fig_insight.update_yaxes(showgrid=True,gridcolor="rgba(0,0,0,0.08)")

    st.plotly_chart(fig_insight,
        use_container_width=True,
        config={"displayModeBar": False})

# KEY BUSINESS FINDINGS

    st.subheader("Key Business Findings")
    col1, col2 = st.columns(2)

    with col1:
        st.success(
            f"Price Performance\n\n"
            f"NVIDIA moved from ${harga_awal_insight:,.2f} "
            f"to ${harga_akhir_insight:,.2f}, "
            f"a change of {persentase_insight:+,.2f}%.")

        st.info(
            f"Price Range\n\n"
            f"Highest close: ${harga_tertinggi_insight:,.2f}\n\n"
            f"Lowest close: ${harga_terendah_insight:,.2f}")

    with col2:
        st.warning(
            f"Risk Profile\n\n"
            f"Annualized volatility: "
            f"{volatilitas_tahunan:,.2f}% "
            f"({kategori_risiko}).")

        st.info(
            f"Trading Activity\n\n"
            f"Average daily volume: "
            f"{rata_rata_volume_insight / 1_000_000:,.2f}M shares.")

    # BUSINESS INTERPRETATION

    st.subheader("Business Interpretation")
    st.write(f"- Trend: {status_deskripsi}")
    st.write(f"- Risk: {kategori_risiko}.")
    st.write(
        f"- Price Position: Latest close "
        f"${harga_akhir_insight:,.2f}, "
        f"average close ${rata_rata_close_insight:,.2f}.")

    # LIMITATION

    st.subheader("Limitation")
    st.write("- Analysis is based on historical price and volume.")
    st.write("- News and macroeconomic factors are not included.")
    st.warning("For educational data-analysis purposes only. ""Not investment advice.")


    # py -m streamlit run nvidia.py



