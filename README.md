# Deteksi Penyakit Daun Jagung - AI System

Sistem deteksi penyakit daun jagung berbasis AI menggunakan ResNet-50 untuk mengklasifikasikan 3 kelas:
- **Hawar**: Daun jagung terkena penyakit hawar
- **Sehat**: Daun jagung dalam kondisi sehat  
- **Karat**: Daun jagung terkena penyakit karat

## Struktur Project

```
corn_leaf_ai/
│
├── app.py                      # Flask backend application
├── requirements.txt            # Dependencies Python
├── README.md                   # Dokumentasi project
│
├── model/
│   └── model_resnet50.pth     # Model ResNet-50 yang sudah ditraining
│
├── static/
│   ├── style.css              # Styling CSS
│   └── uploads/               # Folder untuk menyimpan gambar upload
│
├── templates/
│   ├── index.html             # Halaman upload gambar
│   └── result.html            # Halaman hasil prediksi
│
└── notebook/
    ├── train_resnet50.ipynb   # Jupyter Notebook untuk training model
    └── dataset/               # Dataset untuk training
        ├── train/
        │   ├── hawar/
        │   ├── sehat/
        │   └── karat/
        └── val/
            ├── hawar/
            ├── sehat/
            └── karat/
```

## Instalasi

1. **Clone atau download project ini**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Training Model (Opsional):**
   - Buka file `notebook/train_resnet50.ipynb`
   - Pastikan dataset sudah ada di folder `dataset/` dengan struktur:
     - `Hawar/` - gambar daun hawar
     - `Sehat/` - gambar daun sehat
     - `Karat/` - gambar daun karat
   - Jalankan semua cell di notebook untuk training model selama 50 epoch
   - Model akan tersimpan di `model/model_resnet50.pth`

4. **Jalankan Aplikasi Flask:**
```bash
python app.py
```

5. **Buka browser dan akses:**
```
http://localhost:5000
```

## Cara Menggunakan

1. Buka aplikasi di browser
2. Klik area upload atau drag & drop gambar daun jagung
3. Klik tombol "Deteksi Sekarang"
4. Lihat hasil prediksi dengan confidence score

## Fitur

- ✅ Frontend HTML murni (tanpa framework)
- ✅ UI modern dengan tema putih-biru muda
- ✅ Preview gambar otomatis sebelum upload
- ✅ Drag & drop support
- ✅ Backend Flask untuk inference
- ✅ Model ResNet-50 dengan akurasi tinggi
- ✅ Menampilkan confidence score
- ✅ Responsive design

## Training Model

Untuk training model sendiri, gunakan notebook `notebook/train_resnet50.ipynb` yang berisi:
- Preprocessing dan data augmentation
- Training ResNet-50 selama 50 epoch
- Evaluasi dengan confusion matrix
- Visualisasi training history
- Unit test untuk prediksi

## Teknologi

- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Backend**: Flask (Python)
- **AI Model**: ResNet-50 (PyTorch)
- **Image Processing**: PIL, torchvision

## Catatan

- Pastikan model `model_resnet50.pth` sudah ada sebelum menjalankan aplikasi
- Format gambar yang didukung: JPG, JPEG, PNG
- Ukuran maksimal file: 16MB
- Model menggunakan preprocessing ImageNet standard

## Lisensi

Project ini dibuat untuk keperluan akademik dan pembelajaran.

