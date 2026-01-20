"""
Kode yang sudah diperbaiki untuk meningkatkan akurasi model
Copy-paste bagian yang relevan ke notebook
"""

# ============================================
# CELL 6: Load Model (IMPROVED)
# ============================================

# Load ResNet-50 pretrained
model = models.resnet50(pretrained=True)

# Ganti fully connected layer untuk 3 kelas
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)

# STRATEGI FINE-TUNING YANG LEBIH BAIK:
# Freeze layer awal (layer1, layer2) - fitur dasar seperti edges, textures
# Unfreeze layer akhir (layer3, layer4) - fitur spesifik untuk dataset kita
# Unfreeze fully connected layer

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

# ============================================
# CELL 7: Setup Optimizer (IMPROVED)
# ============================================

# Loss function
criterion = nn.CrossEntropyLoss()

# DIFFERENTIAL LEARNING RATE:
# Layer akhir (layer3, layer4, fc) menggunakan LR lebih tinggi
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

# ============================================
# CELL 3: Transformasi (IMPROVED AUGMENTATION)
# ============================================

# ImageNet normalization parameters
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

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

# Transformasi untuk validation (tanpa augmentasi)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

print("Transformasi dengan augmentasi yang lebih baik sudah didefinisikan!")

