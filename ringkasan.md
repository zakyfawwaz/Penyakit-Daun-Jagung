# Ringkasan Preprocessing dan Data Augmentation
## Before & After Analysis

---

## 📋 Daftar Isi
1. [Preprocessing - Before & After](#1-preprocessing---before--after)
2. [Data Augmentation - Before & After](#2-data-augmentation---before--after)
3. [Perbandingan Training vs Validation Transform](#3-perbandingan-training-vs-validation-transform)

---

## 1. Preprocessing - Before & After

### 1.1 Gambar Original (BEFORE)
**Kondisi Awal:**
- **Ukuran**: Variabel (contoh: 2850x2850 pixels)
- **Format**: RGB (Red, Green, Blue)
- **Range Pixel**: 0-255 (integer)
- **Type**: PIL Image atau numpy array
- **Status**: Belum siap untuk model

**Karakteristik:**
- Gambar dalam ukuran asli dari dataset
- Belum dinormalisasi
- Belum diubah ke format tensor

---

### 1.2 Resize ke 224x224 (AFTER - Step 1)
**Transformasi:**
```python
transforms.Resize((224, 224))
```

**Perubahan:**
- **BEFORE**: Ukuran variabel (contoh: 2850x2850 pixels)
- **AFTER**: Ukuran tetap 224x224 pixels
- **Method**: Bilinear interpolation
- **Alasan**: ResNet-50 membutuhkan input ukuran 224x224

**Visualisasi:**
- Gambar di-resize ke ukuran standar
- Proporsi gambar mungkin berubah (jika aspect ratio berbeda)
- Detail gambar tetap terlihat jelas

---

### 1.3 ToTensor (AFTER - Step 2)
**Transformasi:**
```python
transforms.ToTensor()
```

**Perubahan:**
- **BEFORE**: PIL Image (format HWC - Height, Width, Channel)
- **AFTER**: PyTorch Tensor (format CHW - Channel, Height, Width)
- **Range**: 0.0 - 1.0 (float32)
- **Shape**: (3, 224, 224) untuk gambar RGB

**Karakteristik:**
- Konversi dari PIL Image ke tensor
- Normalisasi otomatis ke range 0-1
- Siap untuk operasi normalisasi berikutnya

---

### 1.4 Normalisasi ImageNet (AFTER - Step 3)
**Transformasi:**
```python
transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                     std=[0.229, 0.224, 0.225])
```

**Perubahan:**
- **BEFORE**: Range 0.0 - 1.0
- **AFTER**: Range ~-2.5 hingga ~2.5 (normalized)
- **Formula**: `(pixel - mean) / std`
- **Mean**: [0.485, 0.456, 0.406] (untuk R, G, B)
- **Std**: [0.229, 0.224, 0.225] (untuk R, G, B)

**Statistik Normalisasi:**
- **BEFORE Normalization:**
  - Mean: ~0.51
  - Std: ~0.21
  - Range: 0.0 - 1.0

- **AFTER Normalization:**
  - Mean: ~0.0 (mendekati 0)
  - Std: ~1.0 (mendekati 1)
  - Range: ~-2.5 hingga ~2.5

**Alasan:**
- ResNet-50 pretrained dilatih dengan normalisasi ImageNet
- Meningkatkan konvergensi training
- Menyamakan distribusi data dengan data training ImageNet

---

### 1.5 Final Preprocessing (READY FOR MODEL)
**Kondisi Final:**
- **Shape**: (3, 224, 224)
- **Type**: torch.Tensor
- **Range**: Normalized dengan ImageNet mean/std
- **Status**: Siap untuk input ke model ResNet-50

**Ringkasan Preprocessing:**
```
Original Image (2850x2850) 
    ↓ Resize
224x224 Image 
    ↓ ToTensor
Tensor (3, 224, 224) range [0, 1]
    ↓ Normalize
Normalized Tensor (3, 224, 224) range [-2.5, 2.5]
    ↓ READY
Input untuk Model ResNet-50
```

---

## 2. Data Augmentation - Before & After

### 2.1 Gambar Base (BEFORE Augmentation)
**Kondisi:**
- Gambar sudah di-resize ke 256x256
- Siap untuk augmentasi
- Masih dalam format PIL Image

---

### 2.2 RandomResizedCrop (AFTER - Step 1)
**Transformasi:**
```python
transforms.RandomResizedCrop(224, scale=(0.7, 1.0))
```

**Perubahan:**
- **BEFORE**: Gambar 256x256
- **AFTER**: Random crop 224x224 dengan scale 0.7-1.0
- **Efek**: 
  - Variasi posisi crop
  - Variasi ukuran crop (70%-100% dari gambar)
  - Meningkatkan variasi data

**Manfaat:**
- Mencegah overfitting
- Meningkatkan generalisasi model
- Simulasi variasi sudut pandang

---

### 2.3 RandomHorizontalFlip (AFTER - Step 2)
**Transformasi:**
```python
transforms.RandomHorizontalFlip(p=0.5)
```

**Perubahan:**
- **BEFORE**: Gambar normal
- **AFTER**: Gambar ter-flip horizontal (50% probabilitas)
- **Efek**: Cerminan horizontal gambar

**Manfaat:**
- Meningkatkan variasi data 2x
- Simulasi variasi orientasi
- Tidak mengubah makna penyakit (daun bisa dilihat dari kiri/kanan)

---

### 2.4 RandomVerticalFlip (AFTER - Step 3)
**Transformasi:**
```python
transforms.RandomVerticalFlip(p=0.3)
```

**Perubahan:**
- **BEFORE**: Gambar normal
- **AFTER**: Gambar ter-flip vertikal (30% probabilitas)
- **Efek**: Cerminan vertikal gambar

**Manfaat:**
- Variasi orientasi tambahan
- Simulasi variasi posisi daun
- Meningkatkan robust model

---

### 2.5 RandomRotation (AFTER - Step 4)
**Transformasi:**
```python
transforms.RandomRotation(degrees=20)
```

**Perubahan:**
- **BEFORE**: Gambar lurus
- **AFTER**: Gambar dirotasi ±20 derajat (random)
- **Efek**: Rotasi gambar dengan sudut random

**Manfaat:**
- Simulasi variasi sudut pengambilan foto
- Meningkatkan invarian rotasi model
- Mencegah overfitting pada orientasi tertentu

---

### 2.6 ColorJitter (AFTER - Step 5)
**Transformasi:**
```python
transforms.ColorJitter(brightness=0.3, contrast=0.3, 
                       saturation=0.3, hue=0.1)
```

**Perubahan:**
- **BEFORE**: Warna original
- **AFTER**: Variasi warna dengan parameter:
  - **Brightness**: ±30% variasi kecerahan
  - **Contrast**: ±30% variasi kontras
  - **Saturation**: ±30% variasi saturasi
  - **Hue**: ±10% variasi hue

**Manfaat:**
- Simulasi variasi pencahayaan
- Simulasi variasi kondisi kamera
- Meningkatkan robust terhadap perubahan warna
- Penting untuk deteksi penyakit (warna bisa bervariasi)

---

### 2.7 RandomAffine (AFTER - Step 6)
**Transformasi:**
```python
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), 
                        scale=(0.9, 1.1))
```

**Perubahan:**
- **BEFORE**: Gambar di posisi tengah
- **AFTER**: Transformasi affine dengan:
  - **Translation**: ±10% pergeseran horizontal/vertikal
  - **Scale**: 90%-110% zoom in/out
  - **Rotation**: 0 derajat (sudah ada RandomRotation terpisah)

**Manfaat:**
- Simulasi variasi posisi objek
- Simulasi variasi jarak pengambilan foto
- Meningkatkan invarian translasi dan scale

---

### 2.8 RandomErasing (AFTER - Step 7)
**Transformasi:**
```python
transforms.RandomErasing(p=0.2, scale=(0.02, 0.33))
```

**Perubahan:**
- **BEFORE**: Gambar utuh
- **AFTER**: Random patch dihapus (20% probabilitas)
- **Efek**: Patch random 2%-33% dari gambar dihapus (diisi dengan nilai random)

**Manfaat:**
- Regularisasi tambahan
- Mencegah overfitting
- Meningkatkan robust terhadap noise/occlusion
- Simulasi kondisi gambar tidak sempurna

---

### 2.9 Full Training Transform (FINAL)
**Kombinasi Semua Augmentasi:**
```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, 
                          saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), 
                            scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.33))
])
```

**Hasil:**
- Setiap epoch, gambar yang sama akan terlihat berbeda
- Meningkatkan variasi data secara signifikan
- Mencegah overfitting
- Meningkatkan generalisasi model

**Ringkasan Data Augmentation:**
```
Original Image (256x256)
    ↓ RandomResizedCrop
224x224 (random crop & scale)
    ↓ RandomHorizontalFlip
Flip horizontal (50% chance)
    ↓ RandomVerticalFlip
Flip vertikal (30% chance)
    ↓ RandomRotation
Rotate ±20° (random)
    ↓ ColorJitter
Variasi brightness, contrast, saturation, hue
    ↓ RandomAffine
Translate ±10%, scale 0.9-1.1
    ↓ ToTensor
Convert to tensor
    ↓ Normalize
ImageNet normalization
    ↓ RandomErasing
Erase random patches (20% chance)
    ↓ FINAL
Augmented Image Ready for Training
```

---

## 3. Perbandingan Training vs Validation Transform

### 3.1 Training Transform (DENGAN Augmentation)
**Transformasi Lengkap:**
1. Resize to 256x256
2. RandomResizedCrop 224x224
3. RandomHorizontalFlip (p=0.5)
4. RandomVerticalFlip (p=0.3)
5. RandomRotation (±20°)
6. ColorJitter
7. RandomAffine
8. ToTensor
9. Normalize
10. RandomErasing (p=0.2)

**Karakteristik:**
- ✅ **Dengan Augmentation**: Setiap epoch gambar berbeda
- ✅ **Variasi Tinggi**: Meningkatkan variasi data
- ✅ **Anti Overfitting**: Mencegah model menghafal data
- ✅ **Generalization**: Meningkatkan kemampuan generalisasi

**Tujuan:**
- Meningkatkan variasi data training
- Mencegah overfitting
- Meningkatkan robust model

---

### 3.2 Validation Transform (TANPA Augmentation)
**Transformasi Sederhana:**
```python
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
```

**Karakteristik:**
- ❌ **Tanpa Augmentation**: Gambar selalu sama
- ✅ **Konsisten**: Evaluasi konsisten setiap kali
- ✅ **Reproducible**: Hasil evaluasi dapat direproduksi
- ✅ **Fair Evaluation**: Evaluasi yang adil tanpa variasi random

**Tujuan:**
- Evaluasi yang konsisten
- Mengukur performa model yang sebenarnya
- Tidak ada variasi random yang mempengaruhi hasil

---

### 3.3 Perbandingan Visual

**Training Transform:**
```
Original → Resize 256x256 → RandomCrop 224x224 → 
RandomFlip → RandomRotation → ColorJitter → 
RandomAffine → ToTensor → Normalize → RandomErasing
```
**Hasil**: Gambar berbeda setiap epoch

**Validation Transform:**
```
Original → Resize 224x224 → ToTensor → Normalize
```
**Hasil**: Gambar selalu sama

---

### 3.4 Kesimpulan Perbandingan

| Aspek | Training Transform | Validation Transform |
|-------|-------------------|---------------------|
| **Augmentation** | ✅ Ya (10 transformasi) | ❌ Tidak |
| **Variasi** | ✅ Tinggi | ❌ Tidak ada |
| **Konsistensi** | ❌ Berubah setiap epoch | ✅ Konsisten |
| **Tujuan** | Meningkatkan variasi data | Evaluasi yang adil |
| **Reproducibility** | ❌ Tidak reproducible | ✅ Reproducible |

**Alasan Perbedaan:**
- **Training**: Butuh variasi untuk mencegah overfitting dan meningkatkan generalisasi
- **Validation**: Butuh konsistensi untuk evaluasi yang fair dan reproducible

---

## 📊 Ringkasan Keseluruhan

### Preprocessing Pipeline
```
Original Image (variabel size)
    ↓
Resize 224x224
    ↓
ToTensor (0-1 range)
    ↓
Normalize (ImageNet mean/std)
    ↓
READY FOR MODEL
```

### Data Augmentation Pipeline (Training)
```
Original Image
    ↓
Resize 256x256
    ↓
RandomResizedCrop 224x224
    ↓
RandomHorizontalFlip
    ↓
RandomVerticalFlip
    ↓
RandomRotation
    ↓
ColorJitter
    ↓
RandomAffine
    ↓
ToTensor
    ↓
Normalize
    ↓
RandomErasing
    ↓
AUGMENTED IMAGE
```

### Key Takeaways
1. **Preprocessing** mengubah gambar menjadi format yang siap untuk model
2. **Data Augmentation** meningkatkan variasi data untuk training
3. **Training** menggunakan augmentasi, **Validation** tidak
4. **Normalisasi ImageNet** penting untuk model pretrained
5. Setiap transformasi memiliki tujuan spesifik untuk meningkatkan performa model

---

## 🔍 Detail Teknis

### Normalisasi ImageNet
- **Mean**: [0.485, 0.456, 0.406] untuk R, G, B
- **Std**: [0.229, 0.224, 0.225] untuk R, G, B
- **Formula**: `normalized = (pixel - mean) / std`
- **Range Hasil**: ~-2.5 hingga ~2.5

### Probabilitas Augmentasi
- **RandomHorizontalFlip**: 50% (p=0.5)
- **RandomVerticalFlip**: 30% (p=0.3)
- **RandomErasing**: 20% (p=0.2)
- **Lainnya**: 100% (selalu diterapkan dengan parameter random)

### Ukuran Input
- **Training**: 256x256 → RandomCrop 224x224
- **Validation**: Langsung Resize 224x224
- **Model Input**: 224x224 (standar ResNet-50)

---

**Dokumen ini dibuat berdasarkan analisis dari notebook `train_resnet50.ipynb`**

