# 🚀 Panduan Meningkatkan Akurasi Model

Model saat ini hanya melatih **fully connected layer** saja (6,147 parameter), sehingga akurasi rendah. Berikut perbaikan yang perlu dilakukan:

## ❌ Masalah Saat Ini:
- Hanya FC layer yang dilatih (6,147 parameter)
- Semua layer ResNet-50 di-freeze
- Model tidak bisa belajar fitur spesifik untuk daun jagung

## ✅ Solusi: Fine-tuning yang Lebih Baik

### 1. **Unfreeze Layer Akhir ResNet-50**

Di cell yang memuat model (Cell 6), ganti kode dengan:

```python
# Load ResNet-50 pretrained
model = models.resnet50(pretrained=True)

# Ganti fully connected layer untuk 3 kelas
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)

# STRATEGI FINE-TUNING YANG LEBIH BAIK:
# Freeze layer awal (layer1, layer2) - fitur dasar seperti edges, textures
# Unfreeze layer akhir (layer3, layer4) - fitur spesifik untuk dataset kita

# Freeze layer awal
for param in model.conv1.parameters():
    param.requires_grad = False
for param in model.bn1.parameters():
    param.requires_grad = False
for param in model.layer1.parameters():
    param.requires_grad = False
for param in model.layer2.parameters():
    param.requires_grad = False

# Unfreeze layer akhir untuk fine-tuning
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

# Pindahkan model ke device
model = model.to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print("Model ResNet-50 sudah dimuat dengan fine-tuning strategy!")
print(f"Jumlah parameter trainable: {trainable_params:,}")
print(f"Total parameter: {total_params:,}")
print(f"Persentase trainable: {100*trainable_params/total_params:.2f}%")
```

### 2. **Gunakan Differential Learning Rate**

Di cell setup optimizer (Cell 7), ganti dengan:

```python
# Loss function
criterion = nn.CrossEntropyLoss()

# DIFFERENTIAL LEARNING RATE:
# Layer akhir menggunakan LR berbeda untuk fine-tuning yang lebih baik
layer3_params = list(model.layer3.parameters())
layer4_params = list(model.layer4.parameters())
fc_params = list(model.fc.parameters())

# Optimizer dengan differential learning rate
optimizer = optim.Adam([
    {'params': layer3_params, 'lr': 0.0001},  # LR lebih kecil untuk layer3
    {'params': layer4_params, 'lr': 0.0005},  # LR sedang untuk layer4
    {'params': fc_params, 'lr': 0.001}        # LR lebih besar untuk FC layer
], weight_decay=1e-4)

# Learning rate scheduler dengan cosine annealing untuk konvergensi lebih baik
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

print("Optimizer dengan differential learning rate sudah disetup!")
print("Learning rates:")
print(f"  Layer3: 0.0001")
print(f"  Layer4: 0.0005")
print(f"  FC Layer: 0.001")
```

### 3. **Tingkatkan Data Augmentation**

Di cell transformasi (Cell 3), ganti train_transform dengan:

```python
# Transformasi untuk training (dengan augmentasi yang lebih agresif)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.33))  # Random erasing untuk regularisasi
])
```

## 📊 Hasil yang Diharapkan:

Setelah perbaikan ini:
- **Parameter trainable**: ~10-15 juta (dari 6,147)
- **Akurasi validation**: Diharapkan naik dari ~50-60% menjadi **85-90%+**
- **Confusion matrix**: Lebih sedikit misclassification antara hawar dan karat

## 🔄 Langkah-langkah:

1. Buka notebook `train_resnet50.ipynb`
2. Update Cell 6 (Load Model) dengan kode di atas
3. Update Cell 7 (Setup Optimizer) dengan kode di atas
4. Update Cell 3 (Transformasi) dengan augmentasi yang lebih baik
5. Jalankan ulang training dari awal
6. Model baru akan tersimpan di `model/model_resnet50.pth`

## ⚠️ Catatan:

- Training akan lebih lama karena lebih banyak parameter yang dilatih
- Pastikan GPU tersedia untuk training yang lebih cepat
- Jika menggunakan CPU, pertimbangkan mengurangi batch size atau epoch

