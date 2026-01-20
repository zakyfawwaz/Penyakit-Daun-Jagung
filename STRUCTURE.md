# Struktur Project Deteksi Penyakit Daun Jagung

## 📁 Struktur Folder Lengkap

```
Jagung Manis/
│
├── app.py                          # ✅ Flask backend application
├── inference.py                    # ✅ Script inference standalone
├── requirements.txt                # ✅ Dependencies Python
├── README.md                       # ✅ Dokumentasi project
├── STRUCTURE.md                    # ✅ File ini
│
├── model/                          # ✅ Folder model
│   ├── .gitkeep
│   └── model_resnet50.pth         # ⚠️ Akan dibuat setelah training
│
├── static/                         # ✅ Folder static files
│   ├── style.css                  # ✅ Styling CSS (putih-biru muda)
│   └── uploads/                   # ✅ Folder upload gambar
│       └── .gitkeep
│
├── templates/                      # ✅ Folder HTML templates
│   ├── index.html                 # ✅ Halaman upload (HTML murni)
│   └── result.html                # ✅ Halaman hasil prediksi
│
├── notebook/                       # ✅ Folder notebook training
│   ├── train_resnet50.ipynb       # ✅ Jupyter Notebook training (50 epoch)
│   └── dataset/                   # ✅ Dataset untuk training
│       ├── train/
│       │   ├── hawar/
│       │   ├── sehat/
│       │   └── karat/
│       └── val/
│           ├── hawar/
│           ├── sehat/
│           └── karat/
│
└── dataset/                        # ✅ Dataset utama (sudah ada)
    ├── Hawar/                     # ✅ 107 gambar
    ├── Sehat/                     # ✅ 101 gambar
    └── Karat/                     # ✅ 102 gambar
```

## ✅ Checklist Fitur

### Frontend (HTML Murni)
- [x] `index.html` - Form upload dengan preview otomatis
- [x] `index.html` - Drag & drop support
- [x] `index.html` - Tombol "Deteksi Sekarang"
- [x] `index.html` - UI modern, responsif, warna putih-biru muda
- [x] `index.html` - Tanpa framework (no Bootstrap, no Tailwind)
- [x] `result.html` - Menampilkan gambar hasil upload
- [x] `result.html` - Menampilkan prediksi (hawar/sehat/karat)
- [x] `result.html` - Menampilkan confidence score
- [x] `result.html` - Tombol kembali
- [x] `style.css` - Styling lengkap dengan tema putih-biru muda

### Backend Flask
- [x] `app.py` - Route `/` untuk halaman upload
- [x] `app.py` - Route `/predict` untuk menerima gambar
- [x] `app.py` - Load model ResNet-50
- [x] `app.py` - Preprocessing: Resize 224×224, Normalisasi ImageNet
- [x] `app.py` - Return label prediksi dan confidence

### Jupyter Notebook Training
- [x] `train_resnet50.ipynb` - Import libraries lengkap
- [x] `train_resnet50.ipynb` - Persiapan dataset (train/val split)
- [x] `train_resnet50.ipynb` - Preprocessing & Augmentasi
- [x] `train_resnet50.ipynb` - Load ResNet-50 pretrained
- [x] `train_resnet50.ipynb` - Ganti FC layer untuk 3 kelas
- [x] `train_resnet50.ipynb` - Training 50 epoch (WAJIB)
- [x] `train_resnet50.ipynb` - Optimizer Adam, Loss CrossEntropy
- [x] `train_resnet50.ipynb` - Simpan model ke `model_resnet50.pth`
- [x] `train_resnet50.ipynb` - Evaluasi: Akurasi training & validation
- [x] `train_resnet50.ipynb` - Grafik loss & accuracy
- [x] `train_resnet50.ipynb` - Confusion matrix
- [x] `train_resnet50.ipynb` - Contoh prediksi satu gambar test
- [x] `train_resnet50.ipynb` - Unit test: Load model & prediksi

### File Tambahan
- [x] `requirements.txt` - Dependencies lengkap
- [x] `README.md` - Dokumentasi lengkap
- [x] `inference.py` - Script inference standalone

## 🚀 Cara Menggunakan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Training Model (Opsional)
1. Buka `notebook/train_resnet50.ipynb`
2. Pastikan dataset sudah ada di folder `dataset/`
3. Jalankan semua cell untuk training 50 epoch
4. Model akan tersimpan di `model/model_resnet50.pth`

### 3. Jalankan Aplikasi Flask
```bash
python app.py
```
Buka browser: `http://localhost:5000`

### 4. Inference Standalone (Opsional)
```bash
python inference.py dataset/Hawar/001.jpg
```

## 📊 Model Information

- **Architecture**: ResNet-50
- **Input Size**: 224×224
- **Classes**: 3 (hawar, sehat, karat)
- **Preprocessing**: ImageNet normalization
- **Training Epochs**: 50
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss

## ✨ Fitur UI

- Modern design dengan gradient biru muda
- Responsive untuk mobile dan desktop
- Drag & drop file upload
- Preview gambar sebelum upload
- Animasi smooth dan transisi
- Color-coded prediction labels:
  - 🔴 Hawar (red gradient)
  - 🟢 Sehat (green gradient)
  - 🟠 Karat (orange gradient)

## 📝 Catatan

- Model `model_resnet50.pth` harus ada sebelum menjalankan aplikasi
- Format gambar: JPG, JPEG, PNG
- Ukuran maksimal: 16MB
- Dataset akan otomatis di-split menjadi train/val oleh notebook

