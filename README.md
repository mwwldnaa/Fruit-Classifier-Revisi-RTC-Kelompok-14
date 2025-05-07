# Fruit Classifier Neural Network

## Gambaran Umum

Proyek ini mengimplementasikan jaringan saraf tiruan untuk klasifikasi buah menggunakan Rust untuk fungsi intelijen mesin inti dan antarmuka berbasis Qt untuk interaksi. Sistem dapat mengklasifikasikan buah berdasarkan karakteristik fisiknya (berat, ukuran, lebar, tinggi) dan memberikan umpan balik visual tentang proses pelatihan.

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

## Format Dataset

Sistem memasukkan file CSV dengan 4 fitur input utama berikut:
- `weight`: Berat buah dalam gram
- `size`: Ukuran buah dalam cm
- `width`: Lebar buah dalam cm
- `height`: Tinggi buah dalam cm
- `label`: Kategori buah (contoh: "apple", "orange")

Contoh dataset yang digunakan:
```
150,7,6,6,apple
```

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

Jaringan saraf mencapai akurasi tinggi (biasanya >95% dengan pelatihan yang cukup) pada dataset yang diformat dengan benar. Antarmuka memberikan umpan balik visual tentang progres pelatihan dan hasil akhir.

## Detail Teknis

- **Backend**: Rust dengan ndarray untuk komputasi numerik
- **Frontend**: Qt/C++ dengan widget plotting khusus
- **Antarmuka Cross-language**: FFI kompatibel-C untuk komunikasi Rust-Qt
- **Normalisasi**: Penskalaan fitur otomatis selama pelatihan dan prediksi

## Cara Build

1. Build library Rust:
   ```bash
   cd rust_backend
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


