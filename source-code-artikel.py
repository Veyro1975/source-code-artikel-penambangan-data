# **Langkah 1: Pemuatan dan Inspeksi Data**

# Mengimport Dataset:

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/Veyro1975/dataset-jurnal-penambangan-data/refs/heads/main/covid_19_indonesia_time_series_all.csv')

# Menampilkan 5 baris awal dari tabel

df.head()

# Mengecek informasi dasar tabel

df.info()

# Menampilkan statistik deskriptif kolom numerikal dari tabel

df.describe()

# **Langkah 2: Penanganan Missing Values**

# enghitung jumlah data dengan missing values

df.isnull().sum()

# Menghitung persentase data dengan missing values

df.isnull().sum()/df.shape[0]*100

#Mengecek pendistribusian data

from scipy.stats import shapiro

numerical_cols = df.select_dtypes(include=np.number).columns

print("Shapiro-Wilk Test for Normality:")
for col in numerical_cols:
    stat, p = shapiro(df[col])
    print(f"\nColumn: {col}")
    print(f"  Test Statistic: {stat:.4f}")
    print(f"  P-value: {p:.4f}")

    # Interpret the results (common alpha = 0.05)
    alpha = 0.05
    if p > alpha:
        print(f"  The data in '{col}' appears to be normally distributed (fail to reject H0)")
    else:
        print(f"  The data in '{col}' does not appear to be normally distributed (reject H0)")

# Penanganan missing values pada kolom kategorikal dan numerikal

df.replace("-", np.nan, inplace=True)

df_filled = df.copy()

# Kategorikal
for col in df_filled.select_dtypes(include='object').columns:
  df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])

# Numerikal
for col in df_filled.select_dtypes(include=[np.number]).columns:
  df_filled[col] = df_filled[col].fillna(df_filled[col].median())

print(df_filled.isnull().sum())

# **Langkah 3: Penanganan Outlier**

# Menampilkan boxplot

import warnings
warnings.filterwarnings('ignore')

# Get columns with numerical data and at least one non-missing value
numerical_cols_with_data = df.select_dtypes(include=np.number).columns[df.select_dtypes(include=np.number).notna().any()]

for i in numerical_cols_with_data:
  sns.boxplot(data=df, x=i)
  plt.show()

# Pengaplikasian IQR(Interquartile Range) untuk menangani outlier

def wisker(col):
  q1, q3 = np.percentile(col, [25, 75])
  iqr = q3 - q1
  lower_bound = q1 - (1.5 * iqr)
  upper_bound = q3 + (1.5 * iqr)
  return lower_bound, upper_bound

# Identify numerical columns with data (excluding those with only missing values)
numerical_cols_for_outlier_handling = df.select_dtypes(include=np.number).columns[df.select_dtypes(include=np.number).notna().any()]

# Apply outlier handling to the identified numerical columns
for col in numerical_cols_for_outlier_handling:
  lower_bound, upper_bound = wisker(df[col])
  df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
  df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

# Identify numerical columns with data (excluding those with only missing values)
numerical_cols_for_outlier_handling = df.select_dtypes(include=np.number).columns[df.select_dtypes(include=np.number).notna().any()]

for i in numerical_cols_for_outlier_handling:
  sns.boxplot(df[i])
  plt.show()

# **Langkah 4: Encoding Data Kategorikal**

# Mengidentifikasi kolom yang memiliki tipe object/kategorikal

df.select_dtypes(include='object').columns

# Penerapan Encoding

pd.get_dummies(data=df, columns=["Date", "Location ISO Code", "Location", "Location Level", "Province", "Country", "Continent", "Island", "Time Zone", "Special Status", "Case Fatality Rate", "Case Recovered Rate",], drop_first=True)

# **Langkah 5: Penskalaan Data Numerik**

# Melakukan normalisasi data

df_min_max_manual = df.copy()

# Select only numerical columns for normalization
numerical_cols_for_normalization = df_min_max_manual.select_dtypes(include=np.number).columns

# Apply Min-Max Normalization to numerical columns
for col in numerical_cols_for_normalization:
    min_numeric = df_min_max_manual[col].min()
    max_numeric = df_min_max_manual[col].max()
    # Avoid division by zero if min and max are the same
    if (max_numeric - min_numeric) != 0:
        df_min_max_manual[f'{col}_norm'] = (df_min_max_manual[col] - min_numeric) / (max_numeric - min_numeric)
    else:
        # If all values are the same, normalized value is 0 (or 1, depending on desired output for constant data)
        df_min_max_manual[f'{col}_norm'] = 0

# Display the normalized columns
normalized_cols = [col for col in df_min_max_manual.columns if '_norm' in col]
display(df_min_max_manual[normalized_cols].head())

# Menyimpan dataframe yang telah di-preprocessing

df_filled.to_csv('/content/clean_data.csv', index=False)

# **Langkah 6: Clustering K-Means**

# Select the columns for clustering
clustering_cols = ['Total Cases', 'Total Deaths', 'Total Recovered']
X = df[clustering_cols].copy()

# Scale the data using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled data back to a DataFrame for easier handling
X_scaled = pd.DataFrame(X_scaled, columns=clustering_cols)

# Determine the optimal number of clusters using Elbow Method and Silhouette Score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

inertia = []
silhouette = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot Elbow Method
plt.figure(figsize=(10, 4))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Inertia')
plt.title('Metode Elbow untuk Menentukan Jumlah Cluster Optimal')
plt.show()

# Plot Silhouette Score
plt.figure(figsize=(10, 4))
plt.plot(K, silhouette, 'ro-')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Nilai Silhouette')
plt.title('Nilai Silhouette untuk Berbagai Jumlah Cluster')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaler.fit_transform(df[['Total Cases']]))

print(df.head())


# Group data by cluster and calculate descriptive statistics
# Using median as features might not be normally distributed after outlier handling
cluster_analysis = df.groupby('Cluster')[clustering_cols].median()
print("Median dari Fitur Clustering per Cluster:")
print(cluster_analysis)

# Visualize the clusters
# Using a pairplot to visualize relationships between clustering features colored by cluster
sns.pairplot(df, vars=clustering_cols, hue='Cluster', palette='viridis')
plt.suptitle("Pairplot Fitur Clustering Berdasarkan Cluster", y=1.02)
plt.show()

import folium

# Ambil kolom yang dibutuhkan
data = df[['Location', 'Latitude', 'Longitude', 'Total Cases']].dropna()

# Ambil nilai terakhir per lokasi (agar tidak berulang per tanggal)
data = data.groupby('Location').last().reset_index()

# Standarisasi Total Cases
scaler = StandardScaler()
X = scaler.fit_transform(data[['Total Cases']])

# K-Means Clustering (3 cluster misalnya)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X)

# Buat peta dasar
peta = folium.Map(location=[-2.5, 118], zoom_start=5, tiles='cartodb positron')

# Warna tiap cluster
warna_cluster = {0: 'red', 1: 'orange', 2: 'green'}

# Tambahkan marker ke peta
for _, row in data.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=6,
        color=warna_cluster[row['Cluster']],
        fill=True,
        fill_opacity=0.7,
        popup=f"{row['Location']}<br>Total Cases: {row['Total Cases']:,}<br>Cluster: {row['Cluster']}"
    ).add_to(peta)

# Tampilkan peta
peta

df_min_max_manual.to_csv('/content/clustered_data.csv', index=False)
print("Data dengan label cluster telah disimpan ke clustered_data.csv")