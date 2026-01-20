# ⚡ QUICK FIX - Perbaiki Akurasi Model Sekarang!

## 🚨 Masalah: Model memprediksi daun berpenyakit sebagai "SEHAT" (67.92%)

Ini terjadi karena model hanya melatih **6,147 parameter** (FC layer saja), tidak cukup untuk belajar fitur penyakit daun jagung.

## ✅ SOLUSI CEPAT - Update 3 Cell di Notebook:

### **CELL 6: Load Model** - GANTI DENGAN INI:

```python
# Load ResNet-50 pretrained
model = models.resnet50(pretrained=True)

# Ganti fully connected layer untuk 3 kelas
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)

# ⭐ PERBAIKAN: Unfreeze layer akhir untuk fine-tuning
# Freeze layer awal (fitur dasar)
for param in model.conv1.parameters():
    param.requires_grad = False
for param in model.bn1.parameters():
    param.requires_grad = False
for param in model.layer1.parameters():
    param.requires_grad = False
for param in model.layer2.parameters():
    param.requires_grad = False

# Unfreeze layer akhir (fitur spesifik)
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"✅ Parameter trainable: {trainable:,} (sebelumnya hanya 6,147!)")
print(f"   Total: {total:,} ({100*trainable/total:.1f}% trainable)")
```

### **CELL 7: Setup Optimizer** - GANTI DENGAN INI:

```python
criterion = nn.CrossEntropyLoss()

# ⭐ PERBAIKAN: Differential Learning Rate
layer3_params = list(model.layer3.parameters())
layer4_params = list(model.layer4.parameters())
fc_params = list(model.fc.parameters())

optimizer = optim.Adam([
    {'params': layer3_params, 'lr': 0.0001},
    {'params': layer4_params, 'lr': 0.0005},
    {'params': fc_params, 'lr': 0.001}
], weight_decay=1e-4)

# ⭐ PERBAIKAN: Cosine Annealing (lebih baik dari StepLR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

print("✅ Optimizer dengan differential LR sudah disetup!")
```

### **CELL 3: Transformasi** - GANTI train_transform DENGAN INI:

```python
# ⭐ PERBAIKAN: Augmentasi lebih agresif
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),  # ⭐ TAMBAHAN
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # ⭐ TAMBAHAN
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.33))  # ⭐ TAMBAHAN
])
```

## 📋 LANGKAH-LANGKAH:

1. **Buka notebook** `train_resnet50.ipynb`
2. **Update Cell 6** (Load Model) - copy kode di atas
3. **Update Cell 7** (Optimizer) - copy kode di atas  
4. **Update Cell 3** (Transformasi) - copy train_transform di atas
5. **Restart kernel** dan jalankan ulang dari awal
6. **Training 50 epoch** - akurasi akan jauh lebih baik!

## 🎯 HASIL YANG DIHARAPKAN:

- ✅ Parameter trainable: **~10-15 juta** (dari 6,147)
- ✅ Akurasi validation: **85-90%+** (dari ~50-60%)
- ✅ Model bisa membedakan hawar, sehat, karat dengan benar
- ✅ Confidence score lebih tinggi dan akurat

## ⚠️ PENTING:

- **Hapus model lama** di `model/model_resnet50.pth` sebelum training baru
- Training akan lebih lama (karena lebih banyak parameter)
- Pastikan dataset sudah di-split dengan benar (train/val)

---

**Setelah training selesai, model baru akan jauh lebih akurat!** 🚀

