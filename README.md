# Laporan Proyek Machine Learning - Guruh Sukmo
## Domain Proyek
Domain yang dipilih untuk proyek *machine learning* ini adalah **Manufaktur**, dengan judul **Predictive Analytics: Machine Maintenance**  

### Latar Belakang

Foto Mesin Miling CNC ![image](https://github.com/user-attachments/assets/6e7de2d0-e62e-44a1-b110-b04ba786119a)


Dalam industri manufaktur, kegagalan mesin mailing secara mendadak dapat menyebabkan kerugian besar baik dari segi waktu, biaya, maupun kualitas produk. Oleh karena itu, predictive maintenance (pemeliharaan prediktif) menjadi sangat penting untuk memprediksi kapan suatu mesin kemungkinan akan gagal, sehingga tindakan pencegahan bisa dilakukan lebih awal.
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
### EDA - Deskripsi Variabel
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
Data yang digunakan dalam pembuatan model merupakan data primer, data ini didapat dari sebuah perusahaan pertanian Amerika, yang disediakan secara publik di kaggle dengan nama datasets yaitu: _Apple Quality_

| Index | UDI  | Product ID | Type | Air Temp (K) | Process Temp (K) | Speed (rpm) | Torque (Nm) | Tool Wear (min) | Machine Failure | TWF | HDF | PWF | OSF | RNF |
| ----- | ---- | ---------- | ---- | ------------ | ---------------- | ----------- | ----------- | --------------- | --------------- | --- | --- | --- | --- | --- |
| 4733  | 4734 | L51913     | L    | 303.3        | 311.6            | 1484        | 44.6        | 223             | 0               | 0   | 0   | 0   | 0   | 0   |
| 2133  | 2134 | H31547     | H    | 299.3        | 309.0            | 1732        | 29.7        | 141             | 0               | 0   | 0   | 0   | 0   | 0   |
| 5807  | 5808 | M20667     | M    | 301.4        | 310.9            | 1489        | 38.9        | 179             | 0               | 0   | 0   | 0   | 0   | 0   |
| 684   | 685  | L47864     | L    | 297.7        | 309.0            | 1538        | 43.2        | 30              | 0               | 0   | 0   | 0   | 0   | 0   |


Tabel 1. EDA Deskripsi Variabel

Dilihat dari _Tabel 1. EDA Deskripsi Variabel_ dataset ini telah di *bersihkan* dan *normalisasi* terlebih dahulu oleh pembuat, sehingga mudah digunakan dan ramah bagi pemula. 
- Dataset berupa CSV (Comma-Seperated Values).
- Dataset memiliki 1000 sample dengan 14 fitur.
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


Dari ke 14 fitur dapat dilihat bahwa fitur `Index` dan `UDI` tidak mempengaruhi kualitas buah hingga akan di hapus.


### Feature Engineering
Kemudian dilakukan Feature Engineering bertujuan memodifikasi fitur (kolom) dalam dataset supaya lebih informatif dan bisa membantu model machine learning bekerja lebih baik.

**Membuat Kolom Baru Yaitu (Temperature Difference dan Mechanical Power)**.
- Kedua fitur ini memperkuat konteks teknis dari data, sehingga model machine learning bisa:
- Belajar pola dengan lebih akurat
- Mendeteksi jenis kegagalan dengan lebih halus
- Meningkatkan performa prediksi karena informasi tambahannya relevan secara fisik dan teknikal

### EDA - Univariate Analysis

![Univariate Analysis] ![image](https://github.com/user-attachments/assets/cfd9802e-ffde-4ae9-ba28-50a825ee6c90)

Gambar 1a. Analisis Univariat (Data Numerik) 

Berdasarkan _Gambar 1a_ , Gambar ini menunjukkan **distribusi dari beberapa fitur numerik** dalam sebuah dataset manufaktur atau industri berbasis sensor. Setiap subplot adalah histogram (dengan garis KDE – *Kernel Density Estimate*) yang menampilkan sebaran data untuk masing-masing fitur. Berikut adalah penjelasan untuk setiap bagian:

---
### 1. **Type**
* **Nilai:** Diskrit (0, 1, dan 2)
* **Penjelasan:** Menandakan jenis mesin atau proses.
* **Distribusi:** Mayoritas data berada pada Type = 1; Type = 0 dan 2 relatif sedikit.
---
### 2. **Air Temperature \[K]**
* **Satuan:** Kelvin
* **Penjelasan:** Temperatur udara sekitar sistem.
* **Distribusi:** Hampir normal tetapi agak miring ke kanan, berkisar antara 296–304 K.
---
### 3. **Process Temperature \[K]**
* **Satuan:** Kelvin
* **Penjelasan:** Temperatur proses saat berjalan.
* **Distribusi:** Cenderung normal, sekitar 308–312 K, pusat sekitar 310 K.
---
### 4. **Rotational Speed \[rpm]**
* **Satuan:** Revolutions per minute
* **Penjelasan:** Kecepatan putar mesin atau alat.
* **Distribusi:** Hampir normal, dengan puncak sekitar 1450 rpm.
---
### 5. **Torque \[Nm]**
* **Satuan:** Newton meter
* **Penjelasan:** Torsi atau gaya puntir.
* **Distribusi:** Agak miring ke kanan, banyak nilai di antara 30–50 Nm.
---
### 6. **Tool Wear \[min]**
* **Satuan:** Menit
* **Penjelasan:** Lama pemakaian alat sebelum diganti.
* **Distribusi:** Hampir seragam antara 0–250 menit, menunjukkan distribusi data seimbang.
---
### 7. **Temperature Difference**

* **Penjelasan:** Selisih antara temperatur proses dan temperatur udara.
* **Distribusi:** Ada dua puncak (bimodal), menunjukkan dua jenis perilaku berbeda pada mesin atau proses.
---
### 8. **Mechanical Power \[W]**
* **Satuan:** Watt
* **Penjelasan:** Daya mekanik yang digunakan.
* **Distribusi:** Hampir normal, puncak sekitar 6000 W, menunjukkan penggunaan energi dominan pada level tersebut.
---
### **Kesimpulan Umum:**

* Data numerik sebagian besar memiliki distribusi normal atau mendekati normal.
* Fitur seperti `Tool wear` dan `Type` menunjukkan perilaku berbeda (diskrit atau hampir seragam).
* Beberapa distribusi (seperti `temperature_difference`) menunjukkan **pola bimodal**, yang bisa mengindikasikan **dua kelompok proses atau kondisi mesin** yang berbeda.

 
### EDA - Multivariate Analysis

![Multivariate Analysis] ![image](https://github.com/user-attachments/assets/7bc962af-6b2e-4ab2-86e6-4dc69ef41b29)

Gambar 2a. Analisis Multivariat

Pada _Gambar 2a. Analisis Multivariat_, dengan menggunakan fungsi _pairplot_
Insight dari Pairplot:

---
### 1. Torque vs. Rotational Speed
- Ada pola invers eksponensial: ketika torque naik, rotational speed turun.
- Kerusakan mesin (oranye) banyak terjadi pada kombinasi torque tinggi dan speed rendah.

### 2. Torque vs. Mechanical Power
- Hubungan linear kuat.
- Kerusakan mesin terjadi pada kombinasi power tinggi dan torque tinggi.

### 3. Rotational Speed vs. Mechanical Power
- Pola eksponensial: saat speed tinggi, power rendah; saat speed rendah, power naik.
- Kerusakan mesin sering muncul pada rotational speed rendah dan power tinggi.

### 4. temperature_difference
- Distribusi bimodal (dua puncak), konsisten dengan sebelumnya.
- Kerusakan mesin muncul di kedua puncak, tapi lebih banyak di sisi kanan (temp_diff tinggi).


## Data Preparation
Pembersihan dan transformasi data sebelum training model.
### Feature Selection 
Proses ini memilih subset fitur (variabel, atribut) yang paling relevan dari dataset untuk digunakan dalam pembangunan model machine learning. Tujuannya adalah untuk meningkatkan kinerja model dan mengurangi kompleksitas.
![image](https://github.com/user-attachments/assets/09d54be8-88d9-4ae4-a0af-8a73cbb8c68a)
Gambar 3a. Feature Selection

Gambar 3a. Kegagalan akibat keausan alat (Tool Wear Failure/TWF), kegagalan karena panas berlebih (Heat Dissipation Failure/HDF), kegagalan daya (Power Failure/PWF), kegagalan akibat beban berlebih (Overstrain Failure/OSF), dan kegagalan acak (Random Failure/RNF) menunjukkan korelasi yang lebih kuat (positif) dengan variabel target, yaitu kegagalan mesin (Machine Failure). Oleh karena itu, kolom 'TWF', 'HDF', 'PWF', 'OSF', dan 'RNF' dihapus.
Karena informasi dari kolom-kolom tersebut sudah "terwakili" atau terlalu erat hubungannya dengan Machine Failure, maka dihapus agar tidak menyebabkan multikolinearitas atau redundansi dalam model prediksi.

### Menghapus Outliers menggunakan Metode IQR
Menghapus outliers yang ada pada dataset
Pada kasus ini, kita akan mendeteksi outliers dengan teknik visualisasi data (boxplot). Kemudian, menangani outliers dengan teknik IQR method.
IQR = Inter Quartile Range IQR = Q3 - Q1
![image](https://github.com/user-attachments/assets/52280862-bb29-419b-a1e3-1db7a38c6a9e)
Gambar 3b.

Gambar 3b. Analisis Boxplot Tiap Fitur:

---
### Air Temperature [K]
- Distribusi simetris, tanpa outlier.
- Rentang sekitar 295 – 305 K.
---
### Process Temperature [K]
- Distribusi normal, tidak ada outlier.
- Sedikit lebih tinggi dari suhu udara.
---
### Rotational Speed [rpm]
- Ada beberapa outlier di sisi atas (sekitar >1850 rpm).
- Data dominan antara 1200–1800 rpm.
---
### Torque [Nm]
- Hampir simetris, 1 outlier di sisi atas.
- Nilai berkisar antara 20–70 Nm.
---
### Tool Wear [min]
- Distribusi menyebar luas (0–250 menit).
- Tanpa outlier, tapi menunjukkan variasi tinggi.
---
### Machine Failure
- Variabel biner: 0 (tidak gagal), 1 (gagal).
- Mayoritas nilai = 0, terlihat sebagai pencilan untuk nilai 1 (failure jarang terjadi).
---
### Temperature Difference
- Hampir tidak ada outlier.
- Sebaran stabil antara 8–12.
---
### Mechanical Power [W]
- Memiliki banyak outlier di atas 9000 W.
- Data utama berada antara 4000–9000 W.
---

## Handling Imbalanced Data menggunakan SMOTE
Tujuan dari handling imbalanced data adalah untuk meningkatkan performa model prediktif, khususnya agar model tidak bias terhadap data minorias
![image](https://github.com/user-attachments/assets/110f91d3-02a3-4a8e-8ae2-fefbaa5c8081)

Gambar 3d. Sebelum SMOTE

---

![image](https://github.com/user-attachments/assets/3e1ec7f2-fa27-4541-bc03-72e9a4600ad8)

Gambar 3e. Setelah SMOTE

## Modeling
Algoritma pada proyek ini melakukan pemodelan dengan berbagai algoritma, yaitu:

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

###  Ringkasan:

* **Cepat & sederhana:** `LogisticRegression`, `SGDClassifier`
* **Akurasi tinggi (tapi lambat):** `GradientBoostingClassifier`, `SVC`
* **Stabil & andal secara umum:** `RandomForestClassifier`
* **Untuk baseline/testing awal:** `DecisionTreeClassifier`, `KNN`

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

Di sini, kita dapat melihat dengan jelas bahwa Random Forest memberikan akurasi terbaik dibandingkan semua algoritma. Oleh karena itu, kita akan melakukan penyetelan parameter lebih lanjut untuk mendapatkan performa yang lebih baik.

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
