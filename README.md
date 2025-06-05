# Laporan Proyek Machine Learning - Guruh Sukmo
## Domain Proyek
Domain yang dipilih untuk proyek *machine learning* ini adalah **Manufaktur**, dengan judul **Predictive Analytics: Machine Maintenance**  

### Latar Belakang

Foto Mesin Miling CNC ![image](https://github.com/user-attachments/assets/6e7de2d0-e62e-44a1-b110-b04ba786119a)


Dalam industri manufaktur, kegagalan mesin miling secara mendadak dapat menyebabkan kerugian besar baik dari segi waktu, biaya, maupun kualitas produk. Oleh karena itu, predictive maintenance (pemeliharaan prediktif) menjadi sangat penting untuk memprediksi kapan suatu mesin kemungkinan akan gagal, sehingga tindakan pencegahan bisa dilakukan lebih awal.
Namun, data nyata untuk predictive maintenance seringkali sulit diperoleh dan dibagikan karena alasan keamanan dan kerahasiaan industri. Untuk itu, dataset sintetik ini dibuat untuk mereplikasi kondisi dunia nyata dalam konteks pemeliharaan prediktif industri.

## Business Understanding
Downtime mesin secara tiba-tiba sangat merugikan perusahaan industri, karena dapat:
- Menghambat produksi,
- Meningkatkan biaya operasional,
- Menurunkan kepuasan pelanggan.

Dengan adanya sistem prediktif, perusahaan bisa:
- Melakukan perawatan hanya saat diperlukan,
- Mengurangi downtime tak terduga,
- Meningkatkan efisiensi dan profitabilitas.

### Problem Statements
1. Bagaimana membangun model klasifikasi untuk memprediksi apakah sebuah mesin akan gagal atau tidak berdasarkan data sensor dan kondisi operasionalnya?
2. Fitur-fitur mana yang paling berpengaruh terhadap kemungkinan kegagalan mesin?
3. Seberapa akurat model yang dikembangkan dalam mendeteksi potensi kegagalan?

### Goals
Tujuan dari proyek ini adalah:
1. Membangun model machine learning yang dapat memprediksi kemungkinan kegagalan mesin berdasarkan 14 fitur sensor dan operasional.
2. Mengevaluasi performa berbagai model klasifikasi.
3. Mengidentifikasi fitur-fitur penting (feature importance) yang memengaruhi prediksi kegagalan.

### Solution Statements
Untuk menyelesaikan permasalahan ini, dilakukan pendekatan sebagai berikut:

- Eksplorasi Dataset: Memahami struktur dan distribusi data, serta menangani ketidakseimbangan kelas (imbalance).
- Preprocessing Data: Normalisasi, encoding, dan handling missing values.
- Eksperimen Model: Menggunakan berbagai algoritma klasifikasi (seperti Random Forest, XGBoost, dll) untuk mengevaluasi performa model.
- Feature Selection & Importance: Menentukan fitur yang paling berkontribusi terhadap prediksi kegagalan.


## Data Understanding

**Informasi Datasets**
| Jenis | Keterangan |
| ------ | ------ |
| Title | Predictive Maintenance Dataset (AI4I 2020) |
| Source | [Kaggle](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) |
| Maintainer | [Stephan Matzka ](https://www.kaggle.com/stephanmatzka) |
| License | CC BY-NC-SA 4.0 |
| Visibility | Publik |
| Tags | _Manufacturing, Tabular, Classification, Model Explainability_ |
| Usability | 8.24 |

Berikut informasi pada dataset: 
Data yang digunakan dalam pembuatan model merupakan data primer, data ini didapat dari sebuah perusahaan manufaktur Amerika, yang disediakan secara publik di kaggle dengan nama datasets yaitu: _Machine Maintenance_

- Dataset berupa CSV (Comma-Seperated Values).
- Dataset memiliki 10000 sample dengan 14 fitur.
- Dataset memiliki 9 fitur bertipe integer, 3 fitur bertipe float64 dan 2 fitur bertipe object.
- Tidak terdapat missing value dalam dataset.
### Variable - variable pada dataset
- `Index` : 'Nomor urut data.
- `UDI` : 'Unique Identifier, identifikasi unik untuk setiap entri.
- `Product ID` : 'Kode produk yang terdiri dari tipe (L, M, H) dan nomor seri.
- `Type` : 'Kualitas produk — L (Low), M (Medium), H (High).
- `Air Temp (K)` : 'Suhu udara dalam satuan Kelvin.
- `Process Temp (K)` : 'Suhu proses dalam satuan Kelvin.
- `Speed (rpm)` : 'Kecepatan putar mesin (rotational speed) dalam RPM.
- `Torque (Nm)` : 'Torsi saat proses berlangsung, dalam satuan Newton meter.
- `Tool Wear (min)` : 'Lama pemakaian alat (tool) dalam menit.
- `Machine Failure` : 'Apakah terjadi kegagalan mesin (1 = Ya, 0 = Tidak).
- `TWF` : 'Tool Wear Failure — kegagalan karena keausan alat.
- `HDF` : 'Heat Dissipation Failure — kegagalan karena panas berlebih.
- `PWF` : 'Power Failure — kegagalan karena masalah daya listrik.
- `OSF` : 'Overstrain Failure — kegagalan karena beban berlebih.
- `RNF` : 'Random Failure — kegagalan acak tanpa penyebab spesifik.

### Missing dan Null Values
Tidak terdapat missing dan null values di dataset ini. kemungkinan sudah dibersihkan oleh pembuat data.

### Distribusi Variabel
Gambar Disribusi Mesin ![image](https://github.com/user-attachments/assets/0d09330e-668c-4670-8e19-10bee1047c64)
Gambar 1a. Distribusi Mesin 
Tipe L : Paling banyak digunakan, dengan jumlah sekitar 6000 unit.
Tipe M : Digunakan sekitar 3000 unit.
Tipe H : Paling sedikit digunakan, sekitar 1000 unit.

Gambar Distribusi Kegagalan Mesin diantara Tipe Mesin ![image](https://github.com/user-attachments/assets/f8da3ed2-4951-4141-bb30-38e18244f7dd)

Gambar 1b. Distribusi Kegagalan Mesin terhadap Tipe Mesin
Tipe L memiliki jumlah kerusakan tertinggi secara jumlah, tapi ini bisa disebabkan karena jumlah totalnya memang paling banyak.
Tipe H memiliki jumlah kerusakan paling sedikit, meskipun total populasinya juga kecil.


## Exploratory Data Analysis(EDA)

### Statistik Deskriptif
Fitur	dan Penjelasan Singkat
- Air temperature [K]	Suhu udara rata-rata sekitar 300 K. Nilainya cukup stabil (std ±2 K).
- Process temperature [K]	Suhu proses sedikit lebih tinggi (rata-rata 310 K), juga stabil.
- temperature_difference	Perbedaan suhu antara proses dan udara — rata-rata 10 K.
- Rotational speed [rpm]	Kecepatan putar bervariasi besar (dari 1168 rpm sampai 2886 rpm).
- Torque [Nm]	Torsi rata-rata ~40 Nm, bisa serendah 3.8 Nm atau setinggi 76.6 Nm.
- Tool wear [min]	Lama penggunaan alat — rata-rata 108 menit, maksimum 253 menit.
- Mechanical Power [W]	Daya mekanik rata-rata ~6280 W, menyebar dari 1148 W hingga 10.469 W.

**Failure-related Features (Biner: 0 = tidak gagal, 1 = gagal):**
Machine failure (target utama) hanya 3.39% data yang mengalami kegagalan.
- TWF (Tool Wear Failure): 0.46%
- HDF (Heat Dissipation Failure): 1.15%
- PWF (Power Failure): 0.95%
- OSF (Overstrain Failure): 0.98%
- RNF (Random Failure): 0.19%


### EDA - Univariate Analysis

![Univariate Analysis] ![image](https://github.com/user-attachments/assets/39ae4bef-aca1-4d1f-bed3-51cc8c0055c4)

Gambar 2a. Analisis Univariat (Data Numerik) 

Berdasarkan _Gambar 2a_ , Gambar ini menunjukkan **distribusi dari beberapa fitur numerik** dalam sebuah dataset manufaktur atau industri berbasis sensor. Setiap subplot adalah histogram (dengan garis KDE – *Kernel Density Estimate*) yang menampilkan sebaran data untuk masing-masing fitur. Berikut adalah penjelasan untuk setiap bagian:

---
#### 1. **Air Temperature \[K]**
* **Satuan:** Kelvin
* **Penjelasan:** Temperatur udara sekitar sistem.
* **Distribusi:** Hampir normal tetapi agak miring ke kanan, berkisar antara 296–304 K.
---
#### 2. **Process Temperature \[K]**
* **Satuan:** Kelvin
* **Penjelasan:** Temperatur proses saat berjalan.
* **Distribusi:** Cenderung normal, sekitar 308–312 K, pusat sekitar 310 K.
---
#### 3. **Rotational Speed \[rpm]**
* **Satuan:** Revolutions per minute
* **Penjelasan:** Kecepatan putar mesin atau alat.
* **Distribusi:** Hampir normal, dengan puncak sekitar 1450 rpm.
---
#### 4. **Torque \[Nm]**
* **Satuan:** Newton meter
* **Penjelasan:** Torsi atau gaya puntir.
* **Distribusi:** Agak miring ke kanan, banyak nilai di antara 30–50 Nm.
---
#### 5. **Tool Wear \[min]**
* **Satuan:** Menit
* **Penjelasan:** Lama pemakaian alat sebelum diganti.
* **Distribusi:** Hampir seragam antara 0–250 menit, menunjukkan distribusi data seimbang.
---

### **Kesimpulan Umum:**

* Data numerik sebagian besar memiliki distribusi normal atau mendekati normal.
* Fitur seperti `Tool wear` dan `Type` menunjukkan perilaku berbeda (diskrit atau hampir seragam).

### EDA - Multivariate Analysis

![Multivariate Analysis] ![image](https://github.com/user-attachments/assets/871bf5ca-d069-4df7-ba8c-fc4287b371b6)


Gambar 2a. Analisis Multivariat

Pada _Gambar 2a. Analisis Multivariat_, dengan menggunakan fungsi _pairplot_
Insight dari Pairplot:

#### 1. Distribusi (diagonal):
- Sebagian besar data didominasi oleh label 0 (non-failure).
- Tool wear dan Torque menunjukkan pola distribusi yang cukup menyebar.
- Mesin cenderung gagal (oranye) saat nilai torque dan tool wear tinggi.

#### 2. Korelasi:
- Air temperature dan Process temperature terlihat berkorelasi positif kuat (titik-titik membentuk garis diagonal).
- Torque vs Rotational speed memperlihatkan hubungan non-linear negatif (semakin tinggi torque, rpm cenderung turun).

#### 3. Pemisahan antara failure dan non-failure:

- Pada beberapa fitur (seperti Torque dan Tool wear), titik oranye (failure) cenderung mengumpul di bagian ekstrem atas, menunjukkan bahwa nilai tinggi dari fitur tersebut berkontribusi pada kegagalan mesin.


## Data Preparation
Pembersihan dan transformasi data sebelum training model.

### Drop fitur yang tidak relevan
Dari ke 14 fitur dapat dilihat bahwa fitur `Index` dan `UDI` tidak mempengaruhi kegagalan mesin hingga akan di hapus. Tujuannya untuk menyederhanakan dataset untuk fokus pada fitur yang berkontribusi nyata terhadap prediksi kegagalan mesin.

### Feature Selection 
Proses ini memilih subset fitur (variabel, atribut) yang paling relevan dari dataset untuk digunakan dalam pembangunan model machine learning. Tujuannya adalah untuk meningkatkan kinerja model dan mengurangi kompleksitas.
![image](https://github.com/user-attachments/assets/09d54be8-88d9-4ae4-a0af-8a73cbb8c68a)
Gambar 3a. Feature Selection

Gambar 3a. Kegagalan akibat keausan alat (Tool Wear Failure/TWF), kegagalan karena panas berlebih (Heat Dissipation Failure/HDF), kegagalan daya (Power Failure/PWF), kegagalan akibat beban berlebih (Overstrain Failure/OSF), dan kegagalan acak (Random Failure/RNF) menunjukkan korelasi yang lebih kuat (positif) dengan variabel target, yaitu kegagalan mesin (Machine Failure). Oleh karena itu, kolom 'TWF', 'HDF', 'PWF', 'OSF', dan 'RNF' dihapus.
Karena informasi dari kolom-kolom tersebut sudah "terwakili" atau terlalu erat hubungannya dengan Machine Failure, maka dihapus agar tidak menyebabkan multikolinearitas atau redundansi dalam model prediksi.

### Feature Engineering
Kemudian dilakukan Feature Engineering bertujuan memodifikasi fitur (kolom) dalam dataset supaya lebih informatif dan bisa membantu model machine learning bekerja lebih baik.

**Membuat Kolom Baru Yaitu (Temperature Difference dan Mechanical Power)**.
1. Membuat fitur baru berupa perbedaan suhu antara suhu proses dan suhu udara
`Temperture_difference` = `Process temperature [K]` - `Air temperature [K]`
2. Membuat fitur baru berupa daya mekanik yang dihitung dari torsi dan kecepatan rotasi
`Mechanical Power [W]` = `Torque [Nm]` * `Rotational speed [rpm]`

- Kedua fitur ini memperkuat konteks teknis dari data, sehingga model machine learning bisa:
- Belajar pola dengan lebih akurat
- Mendeteksi jenis kegagalan dengan lebih halus
- Meningkatkan performa prediksi karena informasi tambahannya relevan secara fisik dan teknikal


### Menghapus Outliers menggunakan Metode IQR
Menghapus outliers yang ada pada dataset
Pada kasus ini, kita akan mendeteksi outliers dengan teknik visualisasi data (boxplot). Kemudian, menangani outliers dengan teknik IQR method.
IQR = Inter Quartile Range IQR = Q3 - Q1
![image](https://github.com/user-attachments/assets/52280862-bb29-419b-a1e3-1db7a38c6a9e)
Gambar 3b.

Gambar 3b. Analisis Boxplot Tiap Fitur:

---
#### Air Temperature [K]
- Distribusi simetris, tanpa outlier.
- Rentang sekitar 295 – 305 K.
---
#### Process Temperature [K]
- Distribusi normal, tidak ada outlier.
- Sedikit lebih tinggi dari suhu udara.
---
#### Rotational Speed [rpm]
- Ada beberapa outlier di sisi atas (sekitar >1850 rpm).
- Data dominan antara 1200–1800 rpm.
---
#### Torque [Nm]
- Hampir simetris, 1 outlier di sisi atas.
- Nilai berkisar antara 20–70 Nm.
---
#### Tool Wear [min]
- Distribusi menyebar luas (0–250 menit).
- Tanpa outlier, tapi menunjukkan variasi tinggi.
---
#### Machine Failure
- Variabel biner: 0 (tidak gagal), 1 (gagal).
- Mayoritas nilai = 0, terlihat sebagai pencilan untuk nilai 1 (failure jarang terjadi).
---
#### Temperature Difference
- Hampir tidak ada outlier.
- Sebaran stabil antara 8–12.
---
#### Mechanical Power [W]
- Memiliki banyak outlier di atas 9000 W.
- Data utama berada antara 4000–9000 W.
---

### Normalisasi
**Langkah 1: Mengubah Label Jadi Angka**
Di data asli, ada kolom 'Type', yang berisi jenis mesin: L, M, atau H.
Masalahnya, model machine learning tidak mengerti huruf. Model hanya paham angka.

Jadi, kita pakai LabelEncoder seperti memberi label angka ke setiap jenis mesin:
- L → 0
- M → 1
- H → 2
Sekarang, model bisa membedakan jenis mesin berdasarkan angka.

**Langkah 2: Menyamaratakan Skala Angka**
Lalu, kita lihat ada fitur seperti:
- Rotational speed [rpm]: ribuan angkanya
- Torque [Nm]: puluhan
- Tool wear [min]: ratusan

Jika kita harus mengukur panjang, berat, dan suhu dalam satu alat ukur. Kalau skalanya beda-beda, model bisa bingung dan cenderung "berat sebelah".

Maka kita pakai StandardScaler, yang bekerja seperti menyetel ulang semua angka agar berada dalam rentang dan skala yang seragam (rata-rata jadi 0, standar deviasi jadi 1).
Ini membuat semua fitur punya pengaruh seimbang saat diproses oleh model.

**Hasil Akhir**
Setelah dua langkah ini:
- Variabel kategori 'Type' sudah jadi angka.
- Semua angka fitur sudah distandarisasi.

Data sekarang siap dilatih ke model machine learning seperti layaknya bahan baku yang sudah diproses rapi di lini produksi sebelum masuk ke mesin utama.

### Handling Imbalanced Data menggunakan SMOTE
Tujuan dari handling imbalanced data adalah untuk meningkatkan performa model prediktif, khususnya agar model tidak bias terhadap data minorias
![image](https://github.com/user-attachments/assets/110f91d3-02a3-4a8e-8ae2-fefbaa5c8081)

Gambar 3d. Sebelum SMOTE

---

![image](https://github.com/user-attachments/assets/3e1ec7f2-fa27-4541-bc03-72e9a4600ad8)

Gambar 3e. Setelah SMOTE

### Train-Test Split
Setelah selesai membersihkan dan menyamakan data mesin (dari langkah encoding & scaling sebelumnya) dan menyeimbangkan data. kini saatnya melatih model machine learning dengan membagi data.
- 80% data → diberikan untuk latihan (training) → seperti teknisi belajar dari banyak kasus nyata.
- 20% data → disimpan untuk ujian (testing) → untuk menguji apakah dia benar-benar paham dan bisa memprediksi kondisi baru yang belum pernah dilihat.

  
**Dengan metode ini, kita memastikan:**
- Model belajar dari data yang cukup
- Tapi juga dilatih untuk generalisasi, agar tidak hanya hafal, tapi juga bisa berpikir saat dihadapkan dengan data baru yang nyata.


## Model Development 
Algoritma pada proyek ini melakukan perbandingan model dengan berbagai algoritma, yaitu:

### Kelebihan dan Kekuranan Algoritma yang dipakai
Berikut adalah **perbandingan singkat** dari berbagai model klasifikasi.
| Model                            | Tipe                | Kelebihan                          | Kekurangan                            | Cocok Untuk                        |
| -------------------------------- | ------------------- | ---------------------------------- | ------------------------------------- | ---------------------------------- |
| **LogisticRegression**           | Linear              | Cepat, interpretatif               | Tidak menangani non-linearitas        | Data linier & kecil                |
| **LogisticRegressionCV**         | Linear + CV         | Otomatis cari regularisasi terbaik | Lebih lambat dari LR biasa            | Model regularisasi otomatis        |
| **SGDClassifier**                | Linear              | Cepat untuk data besar/sparse      | Perlu banyak tuning, sensitif         | Teks (bag-of-words), data besar    |
| **DecisionTreeClassifier**       | Non-linear          | Mudah dimengerti, fleksibel        | Overfitting jika tidak dipangkas      | Interpretasi mudah, fitur campuran |
| **RandomForestClassifier**       | Ensemble (bagging)  | Akurat, tahan overfitting          | Lebih lambat, kurang interpretatif    | Data tabular umum                  |
| **GradientBoostingClassifier**   | Ensemble (boosting) | Akurasi tinggi, kontrol fine-tuned | Pelatihan lambat                      | Proyek prediksi presisi tinggi     |
| **AdaBoostClassifier**           | Ensemble (boosting) | Fokus pada kesalahan sebelumnya    | Sensitif terhadap outlier             | Data bersih, sederhana             |
| **BaggingClassifier**            | Ensemble (bagging)  | Kurangi variance, stabil           | Tidak mengurangi bias                 | Cocok untuk model yang overfit     |
| **SVC (Support Vector Machine)** | Non-linear (kernel) | Akurasi tinggi untuk dataset kecil | Skalabilitas rendah, sensitif scaling | Dataset kecil, kompleks            |
| **KNeighborsClassifier (KNN)**   | Lazy                | Tidak perlu pelatihan, intuitif    | Lambat saat prediksi, sensitif skala  | Data kecil, distribusi jelas       |

---

### Cara Kerja setiap Algoritma

**1. Random Forest (Accuracy: 98.17%)**
Cara Kerja:
- Membuat banyak Decision Tree secara acak.
- Setiap pohon hanya melihat sebagian fitur dan data.
- Hasil akhir ditentukan berdasarkan voting mayoritas dari semua pohon.

**2. Bagging (Accuracy: 97.38%)**
Cara Kerja:
- Singkatan dari Bootstrap Aggregating.
- Melatih banyak model (biasanya Decision Tree) pada data acak yang berbeda (dengan pengambilan ulang).
- Outputnya berdasarkan voting mayoritas (klasifikasi) atau rata-rata (regresi).

**3. Decision Tree (Accuracy: 96.44%)**
Cara Kerja:
- Membagi data berdasarkan fitur yang paling "memisahkan" (pakai entropy/gini).
- Setiap node bertanya “ya/tidak” sampai hasil akhir didapat.
Contoh:
"Apakah suhu > 300K?" → Ya: ke kanan, Tidak: ke kiri


**4. Gradient Boosting (Accuracy: 95.39%)**
Cara Kerja:
- Membangun model secara berurutan.
- Setiap model baru mencoba memperbaiki kesalahan model sebelumnya.
- Hasil akhir = kombinasi dari semua model kecil.
- Contoh: Model ke-1 salah memprediksi X → Model ke-2 fokus belajar dari X.

**5. K-Nearest Neighbors (KNN) (Accuracy: 93.12%)**
Cara Kerja:
- Tidak belajar di awal (lazy learner).
- Saat ada data baru, ia mencari K data tetangga terdekat berdasarkan jarak (misal: Euclidean).
- Kelas yang paling banyak → jadi prediksi.

**6. AdaBoost (Accuracy: 92.72%)**
Cara Kerja:
Mirip Gradient Boosting, tapi:
- Model pertama dilatih.
- Data yang salah → diberi bobot lebih tinggi di model selanjutnya.
- Model akhir = gabungan berbobot dari semua model kecil.

**7. Logistic Regression CV (Accuracy: 84.22%)**
Cara Kerja:
- Model linier untuk klasifikasi biner.
- Menggunakan fungsi sigmoid untuk mengubah output jadi probabilitas.
- CV artinya Cross-Validation otomatis untuk memilih parameter terbaik (biasanya C → regularisasi).

**8. Logistic Regression (Accuracy: 83.66%)**
Cara Kerja:
- Sama seperti atas, tapi tanpa cross-validation otomatis.
- Menggunakan satu nilai regularisasi yang ditentukan pengguna/default.

**9. Support Vector Machine (SVC) (Accuracy: 79.99%)**
Cara Kerja:
- Mencari garis/pemisah terbaik (hyperplane) yang memisahkan dua kelas dengan margin terbesar.
- Bisa menggunakan kernel untuk menangani data non-linear.

**10. Stochastic Gradient Descent (SGD) (Accuracy: 58.77%)**
Cara Kerja:

- Model linier yang dioptimasi pakai iterasi per data satu-satu.
- Cepat dan cocok untuk data besar.
- Mirip Logistic Regression tapi pakai metode stochastic (acak).

---


### Perbandingan Hasil Performa Model:

Dalam tahap evaluasi, metrik yang digunakan adalah `accuracy`
Accuracy didapatkan dengan menghitung persentase dari jumlah prediksi yang benar dibagi dengan jumlah seluruh prediksi. Rumus:

Berikut hasil accuracy dari model yang dilatih:
![image](https://github.com/user-attachments/assets/0025c722-8f71-4dbc-8bc1-38d1d82ad70d)

Gambar 4a. Visualisasi Accuracy Model
Kesimpulan Visualisasi Model
- Model Ansambel (Random Forest, Bagging, Gradient Boosting) mendominasi performa tertinggi karena kemampuannya menangani data kompleks dan mengurangi overfitting.
- Model Linier (Logistic Regression, SGD) memiliki performa yang lebih rendah, mungkin karena pola dalam data bersifat non-linier.
- SVM dan KNN bekerja cukup baik, tetapi kalah dari model ansambel karena keterbatasan skalabilitas atau sensitivitas terhadap parameter.

### Memilih Random Forest sebagai Model
Di sini, kita dapat melihat dengan jelas bahwa Random Forest memberikan akurasi terbaik dibandingkan semua algoritma. Oleh karena itu, kita akan melakukan penyetelan parameter lebih lanjut untuk mendapatkan performa yang lebih baik.
Disini, kita menggunakan parameter `class_weight='balanced'` membantu model tetap fokus pada kelas minoritas, meskipun datanya sudah seimbang, model bisa overfitting pada data sintetis (hasil SMOTE) dan tetap bias ke mayoritas.


## Evaluation

Berikut adalah penjelasan dari **classification report** untuk model **Random Forest**:

---

###  **Penjelasan Classification Report – Random Forest**

####  Ringkasan:

* **Akurasi Total**: **0.98**
  Artinya, **98%** dari seluruh data prediksi benar. Ini menunjukkan model sangat andal secara keseluruhan.

---

###  **Per Kelas (Class-wise Analysis)**

| Kelas | Precision | Recall | F1-score | Support | Penjelasan                                                                                                                                                                                                                                                                                                         |
| ----- | --------- | ------ | -------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **0** | 0.99      | 0.98   | 0.98     | 1838    | <ul><li>**Precision 0.99**: Dari semua yang diprediksi sebagai kelas 0, 99% benar.</li><li>**Recall 0.98**: Dari semua data yang benar-benar kelas 0, 98% berhasil diprediksi dengan benar.</li><li>**F1-score 0.98**: Rata-rata harmonis antara precision dan recall, menunjukkan performa sangat baik.</li></ul> |
| **1** | 0.98      | 0.99   | 0.98     | 1871    | <ul><li>**Precision 0.98**: Dari prediksi kelas 1, sebanyak 98% akurat.</li><li>**Recall 0.99**: Hampir semua data kelas 1 dikenali oleh model.</li><li>**F1-score 0.98**: Sangat konsisten dan seimbang.</li></ul>                                                                                                |

---

###  **Rata-Rata (Averaging)**

| Jenis Rata-rata      | Nilai | Penjelasan                                                                                                                                  |
| -------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **Macro Average**    | 0.98  | Rata-rata precision, recall, dan F1-score dari kedua kelas **tanpa mempertimbangkan jumlah data per kelas**. Menunjukkan performa seimbang. |
| **Weighted Average** | 0.98  | Rata-rata yang **memperhitungkan jumlah data di tiap kelas** (support). Ini mencerminkan performa sebenarnya dalam dataset tidak seimbang.  |
| **Support**          | 3709  | Jumlah total sampel yang dievaluasi oleh model (1838 + 1871).                                                                               |

---

###  **Kesimpulan**

* **Random Forest sangat baik** dalam mengklasifikasikan kedua kelas, dengan precision dan recall di atas 98%.
* Tidak ada tanda-tanda ketimpangan prediksi (bias terhadap satu kelas).
* Cocok untuk digunakan dalam **aplikasi nyata** yang memerlukan tingkat akurasi tinggi dan prediksi yang seimbang.

## Feature Importance
Tahap ini untuk mengetahui seberapa besar kontribusi masing-masing fitur (variabel input) terhadap hasil prediksi model machine learning.
![image](https://github.com/user-attachments/assets/3735bebe-e86f-44f3-b95b-2f2fbb4cb6f7)
Gambar 6a. Feature Importance
Interpretasi dari gambar 6a:
- Tool wear [min] memiliki pengaruh terbesar terhadap kegagalan mesin. Hal ini masuk akal, karena keausan alat yang tinggi sangat berkaitan dengan kemungkinan kerusakan mesin.
- Rotational speed [rpm], Torque [Nm], dan Mechanical Power [W] juga merupakan faktor penting yang berkontribusi besar terhadap kegagalan mesin. Ini menunjukkan bahwa parameter operasional mesin sangat memengaruhi kinerjanya.
- temperature_difference juga memiliki pengaruh yang cukup signifikan — ini menunjukkan bahwa perbedaan suhu mungkin berkaitan dengan kondisi abnormal pada mesin.
- Air temperature [K] dan Process temperature [K] memiliki pengaruh yang lebih kecil, namun tetap relevan.
- Type (jenis mesin atau tipe operasi) adalah fitur dengan pengaruh paling kecil terhadap prediksi kegagalan mesin dalam model ini.

## Referensi
1. Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. Kaggle. https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020
2. Purba, J. T., & Supriyanto, E. (2022). Penerapan Algoritma K-Nearest Neighbor (KNN) dalam Klasifikasi Kualitas Buah Apel. Jurnal Teknologi dan Sistem Komputer, 10(1), 33–39.
3. Astuti, D., Prasetyo, D. P., & Widodo, A. (2021). Prediksi Maintenance Mesin Produksi Menggunakan Algoritma Random Forest. Jurnal Informatika dan Sistem Informasi, 7(2), 95–101.
4. Hidayat, R., & Nugroho, A. (2020). Penerapan Support Vector Machine untuk Klasifikasi Citra Apel Berdasarkan Warna dan Tekstur. Jurnal Ilmiah Komputer dan Informatika KOMPUTA, 9(1), 12–19.
5. Putra, A. R., & Lestari, S. D. (2023). Implementasi Algoritma Naive Bayes pada Penentuan Kualitas Buah Apel Menggunakan Citra Digital. Jurnal Penelitian Ilmu Komputer, Sistem Embedded dan Logic, 11(3), 174–180.
6. Saputra, R. A., & Wicaksono, Y. P. (2021). Analisis Prediktif Maintenance Menggunakan Metode Decision Tree dan Naive Bayes. Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi), 5(6), 1098–1105.
_
