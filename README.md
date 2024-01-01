# Laporan Proyek Machine Learning
### Nama : Andhika satria firmansyah
### Nim : 211351154
### Kelas : Pagi A

# Prediksi Iris species menggunakan algoritma KMeans

### Latar belakang
Kumpulan data bunga Iris atau kumpulan data Iris Fisher adalah kumpulan data multivariat yang diperkenalkan oleh ahli statistik Inggris, ahli eugenis, dan ahli biologi Ronald Fisher dalam makalahnya tahun 1936 Penggunaan beberapa pengukuran dalam masalah taksonomi sebagai contoh analisis diskriminan linier.

tiga kelas bunga iris masing-masing terdiri dari 50 sampel. Satu kelas bunga dapat dipisahkan secara linier dari dua lainnya, tetapi dua kelas bunga lainnya tidak dapat dipisahkan secara linier satu sama lain.

## Business understanding

### Problem statements
Karena saya mempunyai pengukuran yang saya tahu spesies iris yang benar, ini adalah masalah pembelajaran yang diawasi. Saya ingin memprediksi salah satu dari beberapa pilihan (spesies iris), menjadikannya contoh masalah klasifikasi. Keluaran yang mungkin (spesies iris yang berbeda) disebut kelas. Setiap iris dalam kumpulan data termasuk dalam salah satu dari tiga kelas yang dipertimbangkan dalam model, jadi masalah ini adalah masalah klasifikasi tiga kelas. Keluaran yang diinginkan untuk satu titik data (sebuah iris) adalah spesies bunga dengan mempertimbangkan fitur-fiturnya. Untuk suatu titik data tertentu, kelas/spesies yang dimilikinya disebut labelnya.

### Goals
Beberapa iris yang sebelumnya telah diidentifikasi oleh ahli botani sebagai spesies setosa, versicolor, atau virginica. Dengan pengukuran ini, dia bisa yakin spesies mana yang dimiliki masing-masing iris. Saya akan mempertimbangkan bahwa ini adalah satu-satunya spesies yang akan ditemui ahli botani.
Tujuannya adalah untuk membuat model pembelajaran mesin yang dapat belajar dari pengukuran iris yang spesiesnya sudah diketahui, sehingga kita dapat memprediksi spesies iris baru yang ditemukan.

### Solution statements
​Menggunakan Algoritma K-Means untuk memprediksi jenis Iris berdasarkan data species,seperti Iris Setosa, Iris Virginica, dan Iris Versicolor.

Algoritma K-Means adalah salah satu algoritma dalam analisis clustering yang digunakan untuk mengelompokkan data ke dalam kategori yang berbeda secara otomatis.

Akurasi: Akurasi adalah ukuran seberapa sering model memprediksi dengan benar.
Sensitivitas: Sensitivitas adalah seberapa sering model memprediksi dengan benar jenis Species Iris.

### Data understanding
Untuk membuat aplikasi Iris species,saya menggunakan dataset "Iris species" dataset ini berisi tentang jenis bunga Iris.
[Iris Species](https://www.kaggle.com/datasets/uciml/iris)  


### Import dataset kaggle

```bash
from google.colab import files
files.upload()
```

```bash
# make directory and change permission
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```

```bash
!kaggle datasets download -d uciml/iris
```

```bash
!unzip iris.zip -d iris1
!ls iris1
```

### Import library yang dibutuhkan

Mengimpor library yang akan digunakan

```bash
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
```

Setelah mengimpor library, dataset yang akan digunakan akan diimpor.

```bash
iris = pd.read_csv("/content/iris1/Iris.csv")
iris.drop('Id',inplace=True,axis=1)
```

### Data discovery

```bash
X = iris.iloc[:,:-1].values

y = iris.iloc[:,-1].values
```

```bash
iris.head().style.background_gradient(cmap =sns.cubehelix_palette(as_cmap=True))
```

![Alt text](1.png)

```bash
iris.info()
```

```bash
iris.sample(5)
```

```bash
iris.describe()
```

### EDA

```bash
fig = px.pie(iris, 'Species',color_discrete_sequence=['#491D8B','#7D3AC1','#EB548C'],title='Data Distribution',template='plotly')

fig.show()
```

![Alt text](2.png)

Datanya sangat seimbang

```bash
fig = px.box(data_frame=iris, x='Species',y='SepalLengthCm',color='Species',color_discrete_sequence=['#29066B','#7D3AC1','#EB548C'],orientation='v')
fig.show()
```

![Alt text](3.png)

```bash
fig = px.histogram(data_frame=iris, x='SepalLengthCm',color='Species',color_discrete_sequence=['#491D8B','#7D3AC1','#EB548C'],nbins=50)
fig.show()
```

![Alt text](4.png)

Setosa memiliki SepalLength yang jauh lebih kecil dibandingkan 2 kelas lainnya

Virginca memiliki SepalLength tertinggi, namun tampaknya sulit membedakan antara Virginca dan Versicolor menggunakan SepalLength karena perbedaannya kurang jelas

Kita dapat melihat bahwa Virginia mengandung outlier

```bash
fig = px.box(data_frame=iris, x='Species',y='SepalWidthCm',color='Species',color_discrete_sequence=['#29066B','#7D3AC1','#EB548C'],orientation='v')
fig.show()
```

![Alt text](5.png)

```bash
fig = px.histogram(data_frame=iris, x='SepalWidthCm',color='Species',color_discrete_sequence=['#491D8B','#7D3AC1','#EB548C'],nbins=30)
fig.show()
```

![Alt text](6.png)

Setosa memiliki SepalWidth yang lebih besar dibandingkan 2 kelas lainnya

Versicolo memiliki SepalWidth yang lebih kecil dibandingkan 2 kelas lainnya

Secara keseluruhan semua kelas tampaknya memiliki nilai sepalwidth yang relatif dekat yang menunjukkan bahwa ini mungkin bukan fitur yang sangat berguna

```bash
fig = px.box(data_frame=iris, x='Species',y='PetalLengthCm',color='Species',color_discrete_sequence=['#29066B','#7D3AC1','#EB548C'],orientation='v')
fig.show()
```

![Alt text](7.png)

```bash
fig = px.histogram(data_frame=iris, x='PetalLengthCm',color='Species',color_discrete_sequence=['#491D8B','#7D3AC1','#EB548C'],nbins=30)
fig.show()
```

![Alt text](8.png)

Setosa memiliki PetaLength yang jauh lebih kecil dibandingkan 2 kelas lainnya

Perbedaan ini kurang jelas antara Virginica dan Versicolor

Secara keseluruhan, ini sepertinya fitur PetaLength yang menarik

```bash
fig = px.box(data_frame=iris, x='Species',y='PetalWidthCm',color='Species',color_discrete_sequence=['#29066B','#7D3AC1','#EB548C'],orientation='v')
fig.show()
```

![Alt text](9.png)

```bash
fig = px.histogram(data_frame=iris, x='PetalWidthCm',color='Species',color_discrete_sequence=['#491D8B','#7D3AC1','#EB548C'],nbins=30)
fig.show()
```

![Alt text](10.png)

Setosa memiliki PetalWidth yang jauh lebih kecil dibandingkan 2 kelas lainnya

Perbedaan ini kurang jelas antara Virginica dan Versicolor

Secara keseluruhan, ini sepertinya fitur PetalWidth yang menarik

```bash
fig = px.scatter(data_frame=iris, x='SepalLengthCm',y='SepalWidthCm'
           ,color='Species',size='PetalLengthCm',template='seaborn',color_discrete_sequence=['#491D8B','#7D3AC1','#EB548C'],)

fig.update_layout(width=800, height=600,
                  xaxis=dict(color="#BF40BF"),
                 yaxis=dict(color="#BF40BF"))
fig.show()
```

![Alt text](11.png)

```bash
fig = px.scatter(data_frame=iris, x='PetalLengthCm',y='PetalWidthCm'
           ,color='Species',size='SepalLengthCm',template='seaborn',color_discrete_sequence=['#491D8B','#7D3AC1','#EB548C'],)

fig.update_layout(width=800, height=600,
                  xaxis=dict(color="#BF40BF"),
                 yaxis=dict(color="#BF40BF"))
fig.show()
```

![Alt text](12.png)

### Data preparation

```bash
iris.head()
```

```bash
iris.columns
```

```bash
iris.isnull().sum()
```

```bash
iris.shape
```

```bash
iris['SepalLengthCm'].unique()
```

```bash
iris['PetalLengthCm'].unique()
```

```bash
iris['PetalWidthCm'].unique()
```

```bash
iris['Species'].unique()
```

```bash
iris = iris.drop('PetalWidthCm', axis = 1)
```

```bash
iris.head().style.background_gradient(cmap = 'RdPu').set_properties(**{'font-family': 'Segoe UI'}).hide_index()
```

![Alt text](13.png)

### Modeling

Menggunakan metode elbow untuk mencari jumlah cluster yang optimal untuk k-means clustering

```bash
sse = []
for i in range(1,9):
    kmeans = KMeans(n_clusters=i , max_iter=300)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

fig = px.line(y=sse,template="seaborn",title='Eblow Method')
fig.update_layout(width=800, height=600,
title_font_color="#BF40BF",
xaxis=dict(color="#BF40BF",title="Clusters"),
yaxis=dict(color="#BF40BF",title="SSE"))
```

![Alt text](14.png)

Seperti yang diharapkan, jumlah cluster yang optimal tampaknya adalah 3 jadi mari kita implementasi model menggunakan 3 cluster

```bash
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
clusters = kmeans.fit_predict(X)
```

### Visualisasi hasil modeling

Sekarang mari kita visualisasikan hasil

```bash
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=X[clusters == 0, 0], y=X[clusters == 0, 1],
    mode='markers',marker_color='#DB4CB2',name='Iris-setosa'
))

fig.add_trace(go.Scatter(
    x=X[clusters == 1, 0], y=X[clusters == 1, 1],
    mode='markers',marker_color='#c9e9f6',name='Iris-versicolour'
))

fig.add_trace(go.Scatter(
    x=X[clusters == 2, 0], y=X[clusters == 2, 1],
    mode='markers',marker_color='#7D3AC1',name='Iris-virginica'
))

fig.add_trace(go.Scatter(
    x=kmeans.cluster_centers_[:, 0], y= kmeans.cluster_centers_[:,1],
    mode='markers',marker_color='#CAC9CD',marker_symbol=4,marker_size=13,name='Centroids'
))
fig.update_layout(template='plotly_dark',width=1000, height=500,title='Kmean Clustering Results')
```

![Alt text](15.png)

### Simpan model pickle

```bash
iris.to_excel("iris.xlsx")
```

### Deployment
[Iris prediction](https://irisspecies-m3qkrun8dxfn8mcfqs9hym.streamlit.app/)
 ![Alt text](img/16.png)<br>
 ![Alt text](img/17.png)<br>
 ![Alt text](img/18.png)<br>
 ![Alt text](img/19.png)<br>