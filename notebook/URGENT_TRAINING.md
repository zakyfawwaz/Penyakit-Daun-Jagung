# 🚨 URGENT: Training Model dengan Akurasi MAKSIMAL

## ⚠️ PENTING: Model Lama Masih Digunakan!

Jika deteksi masih tidak akurat, kemungkinan:
1. **Model lama masih digunakan** di Flask app (`model/model_resnet50.pth`)
2. **Training belum dijalankan** dengan strategi baru
3. **Model perlu di-retrain** dari awal

## ✅ LANGKAH-LANGKAH URGENT:

### 1. HAPUS MODEL LAMA
```bash
# Hapus model lama di folder model/
rm model/model_resnet50.pth
# atau di Windows:
del model\model_resnet50.pth
```

### 2. BUKA NOTEBOOK
- Buka `notebook/train_resnet50.ipynb`
- **RESTART KERNEL** (Kernel → Restart & Clear Output)

### 3. JALANKAN SEMUA CELL DARI AWAL
- Pastikan semua cell dijalankan dengan strategi FULL FINE-TUNING
- Cell 10 akan menunjukkan: "25+ JUTA PARAMETER trainable"
- Cell 12 akan menunjukkan differential LR untuk semua layer

### 4. TRAINING 50 EPOCH
- Training akan lebih lama (karena 25+ juta parameter)
- Tapi akurasi akan SANGAT TINGGI (90-95%+)
- Tunggu sampai selesai!

### 5. PASTIKAN MODEL BARU TERSIMPAN
- Setelah training, model akan tersimpan di `model/model_resnet50.pth`
- Pastikan file ini ada dan baru (cek timestamp)

### 6. RESTART FLASK APP
```bash
# Stop Flask app (Ctrl+C)
# Jalankan lagi:
python app.py
```

## 📊 STRATEGI YANG SUDAH DITERAPKAN:

### ✅ FULL FINE-TUNING
- **SEMUA LAYER UNFROZEN** (25+ juta parameter)
- Bukan hanya FC layer (6,147 parameter)

### ✅ Differential Learning Rate
- Conv1+BN1: 1e-5
- Layer1: 5e-5
- Layer2: 1e-4
- Layer3: 5e-4
- Layer4: 1e-3
- FC: 2e-3

### ✅ Augmentasi Agresif
- RandomVerticalFlip
- RandomAffine
- RandomErasing

### ✅ Scheduler Optimal
- CosineAnnealingWarmRestarts

## 🎯 HASIL YANG DIHARAPKAN:

- ✅ Akurasi: **90-95%+**
- ✅ Model bisa membedakan hawar, sehat, karat dengan benar
- ✅ Confidence score tinggi dan akurat
- ✅ Tidak ada misclassification besar

## ⏰ JIKA DEADLINE SANGAT DEKAT:

Jika waktu sangat terbatas, bisa:
1. Kurangi epoch menjadi 30-40 (tapi tetap FULL FINE-TUNING)
2. Gunakan GPU jika tersedia (akan jauh lebih cepat)
3. Atau gunakan batch size lebih kecil jika memory terbatas

---

**PASTIKAN: Model baru sudah di-train dan digunakan di Flask app!**

