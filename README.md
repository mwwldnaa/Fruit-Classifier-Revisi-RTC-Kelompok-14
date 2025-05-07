# FRUIT CLASSIFIER USING NEURAL NETWORK
# Supervisor:
Ahmad Radhy, S.Si., M.Si

# KELOMPOK 14
1) Juwita Maulidina Mahmudah || 2042221038
2) Dimas Dwi Firmansyah || 2042221050
3) Litakuni Windriya Ramadhani || 2042221119

Departemen Teknik Instrumentasi 
Institut Teknologi Sepuluh Nopember
Surabaya
2025

## Gambaran Umum

Proyek ini mengimplementasikan jaringan saraf tiruan untuk klasifikasi buah menggunakan Rust sebagai logika neural network dan antarmuka berbasis Qt untuk interaksi. Sistem dapat mengklasifikasikan buah berdasarkan karakteristik fisiknya (berat, ukuran, lebar, tinggi) dan memberikan umpan balik visual tentang proses pelatihan.

## Fitur Utama

- **Implementasi Jaringan Saraf Tiruan**:
  - Jaringan 2 lapis dengan aktivasi ReLU dan output softmax
  - Inisialisasi berat Xavier/Glorot
  - Regularisasi L2 untuk mencegah overfitting
  - Dukungan pelatihan batch
  - Fungsi loss cross-entropy

- **Pengolahan Data**:
  - Pembacaan dataset CSV dengan validasi
  - Normalisasi fitur (skala mean/std)
  - One-hot encoding untuk label
  - Fungsi pemisahan data latih/uji

- **Fitur Antarmuka**:
  - Visualisasi progres pelatihan interaktif
  - Grafik akurasi dan loss secara real-time
  - Antarmuka prediksi manual
  - Pemilihan dataset
  - Konfigurasi pelatihan (jumlah epoch)


## Detail Teknis

- **Backend**: Rust dengan ndarray untuk komputasi numerik
- **Frontend**: Qt/C++ dengan widget plotting khusus
- **Antarmuka Cross-language**: FFI kompatibel-C untuk komunikasi Rust-Qt
- **Normalisasi**: Penskalaan fitur otomatis selama pelatihan dan prediksi


## Format Dataset

Sistem memasukkan file CSV dengan 4 fitur input utama berikut:
- `weight`: Berat buah dalam gram
- `size`: Ukuran buah dalam cm
- `width`: Lebar buah dalam cm
- `height`: Tinggi buah dalam cm
- `label`: Kategori buah (contoh: "apple", "orange")

Contoh format dataset yang digunakan:
```
150,7,6,6,apple
```

## Penjelasan Program Rust Backend

* **`data.rs`**:
    * `FruitSample` struct: Mendefinisikan struktur data untuk menyimpan informasi tentang setiap sampel buah, termasuk `weight` (berat), `size` (ukuran), `width` (lebar), `height` (tinggi), dan `label` (jenis buah).
    * `load_dataset` function: Memuat data dari file CSV ke dalam vektor `Vec<FruitSample>`. Fungsi ini juga melakukan validasi dasar terhadap data yang dimuat untuk memastikan tidak ada pengukuran yang tidak valid (misalnya, nilai negatif atau nol).

* **`lib.rs`**:
  Merupakan library yang menyediakan fungsi-fungsi yang dapat dipanggil dari kode C.
    * Structs: Mendefinisikan struktur data untuk `FruitSample`, `Normalizer`, `NeuralNet`, dan `TrainingResult`.
        * `Normalizer`: Digunakan untuk menghitung dan menerapkan normalisasi pada fitur-fitur data.
        * `NeuralNet`: Mengimplementasikan arsitektur dan logika pelatihan jaringan saraf.
        * `TrainingResult`: Digunakan untuk mengembalikan hasil pelatihan.
    * `Normalizer` struct dan metodenya (`fit`, `transform`): Melakukan penskalaan fitur numerik dengan menghitung mean dan standar deviasi dari data pelatihan.
    * `NeuralNet` struct dan metodenya (`new`, `relu`, `softmax`, `forward`, `train_one_epoch`, `cross_entropy_loss`, `evaluate`): Mengimplementasikan jaringan saraf.
        * `new`: Membuat instance `NeuralNet` dengan inisialisasi bobot.
        * `forward`: Melakukan forward pass untuk menghasilkan prediksi.
        * `train_one_epoch`: Melakukan satu epoch pelatihan dengan backpropagation dan pembaruan bobot.
        * `cross_entropy_loss`: Menghitung fungsi loss cross-entropy.
        * `evaluate`: Menghitung akurasi model.
    * Fungsi extern "C" (`train_network`, `predict`, `free_array`, `free_string`): Menyediakan antarmuka untuk berinteraksi dengan library dari kode C, termasuk melatih jaringan, membuat prediksi, dan mengelola memori.

* **`main.rs`**:
  Merupakan program utama yang menjalankan pelatihan model dan menyediakan mode interaktif untuk pengujian manual.
    * `train_model` function:
        * Memuat dataset.
        * Memisahkan data menjadi set pelatihan dan pengujian.
        * Membuat dan melatih model jaringan saraf (`NeuralNet`).
        * Mencetak metrik pelatihan (loss dan akurasi).
    * `plot_training_results` function: Membuat plot visualisasi akurasi dan loss selama proses pelatihan.
    * `main` function:
        * Memanggil `train_model` untuk memulai pelatihan.
        * Memanggil `plot_training_results` untuk menampilkan grafik pelatihan.
        * Memasuki mode pengujian manual, di mana pengguna dapat memasukkan fitur buah untuk mendapatkan prediksi.

* **`model.rs`**:
    * Mendefinisikan struct `NeuralNet` dan method-methodnya.
    * Fungsinya mirip dengan `NeuralNet` di `lib.rs`, tetapi dioptimalkan untuk digunakan dalam program Rust utama.

* **`training.rs`**:
    * Menyediakan fungsi (`run_training_from_csv`, `run_training_from_samples`) untuk menjalankan pelatihan model dari file CSV atau langsung dari data sampel.
    * Mengelola proses pelatihan secara keseluruhan, termasuk pra-pemrosesan data, pembagian dataset, dan penggunaan `NeuralNet`.

* **`utils.rs`**:
    * `Normalizer` struct dan method-methodnya (`new`, `fit`, `transform`, `normalize`): Mengimplementasikan normalisasi fitur.
    * `encode_labels` function: Mengubah label kategori menjadi representasi numerik (one-hot encoding).


## Penjelasan Program Qt Frontend

  * Pada folder **`src/`** terdapat beberap modul dan file dengan masing-masing fungsinya.
      * `main.cpp`:   Fungsi `main` untuk memulai aplikasi Qt.
      * `mainwindow.cpp`/`.h`:  Implementasi kelas `MainWindow` (jendela utama aplikasi).
      * `plotwidget.cpp`/`.h`: Implementasi widget untuk menampilkan grafik (akurasi dan loss).
  
  * `CMakeLists.txt`:  File konfigurasi CMake yang digunakan untuk membangun aplikasi C++.


## Cara Menggunakan

1. **Pelatihan**:
   - Klik "Select Dataset" untuk memilih file CSV
   - Atur jumlah epoch pelatihan
   - Klik "Start Training" untuk memulai proses
   - Lihat plot akurasi dan loss secara real-time

2. **Prediksi**:
   - Masukkan pengukuran buah di kolom input
   - Klik "Predict Fruit" untuk mendapatkan klasifikasi
   - Hasil akan menampilkan jenis buah yang diprediksi

## Kinerja

Jaringan saraf mencapai akurasi tinggi (nilaiakurasi mencapai >95% dengan pelatihan yang cukup) pada dataset yang diformat dengan benar. Antarmuka memberikan umpan balik visual tentang progres pelatihan dan hasil akhir.


## Cara Build

1. Build library Rust:
   ```bash
   cd fruit_classifier/rust_backend
   cargo clean
   cargo build
   cargo run --release (agar terhubung dengan Qt)
   ```

2. Build aplikasi Qt:
   ```bash
   cd fruit_classifier/qt_frontend
   rm -rf build
   mkdir build
   cd build
   cmake ..
   make
   ```

## Persyaratan

- Rust (untuk backend)
- Qt 5+ (untuk GUI)
- Kompiler yang kompatibel dengan C++17

## **Tautan Laporan**:
https://drive.google.com/file/d/1tLRu3Pt5wVn1KsBSouxpTJ_k99KuL1kU/view?usp=drive_link 

## **Tautan PPT**:
https://drive.google.com/file/d/1fpj4M9oWMPnaiqpbhVtdzZYBufHBt0KF/view?usp=drive_link 

