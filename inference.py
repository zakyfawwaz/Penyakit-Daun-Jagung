"""
Script inference untuk prediksi penyakit daun jagung menggunakan ResNet-50
Dapat digunakan secara standalone tanpa Flask
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys

# Model classes - URUTAN HARUS SAMA DENGAN ImageFolder (alfabetis: hawar, karat, sehat)
# ImageFolder mengurutkan kelas secara alfabetis, jadi urutannya: hawar (0), karat (1), sehat (2)
CLASS_NAMES = ['hawar', 'karat', 'sehat']

# ImageNet normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

def load_model(model_path='model/model_resnet50.pth'):
    """Load the trained ResNet-50 model"""
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 3)  # 3 classes
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"Model tidak ditemukan di: {model_path}")

def predict_image(image_path, model):
    """Predict the class of an image"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
            
            predicted_class = CLASS_NAMES[predicted.item()]
            confidence_score = confidence.item() * 100
            
            # Get all probabilities
            all_probs = {CLASS_NAMES[i]: probabilities[i].item() * 100 
                        for i in range(len(CLASS_NAMES))}
            
        return predicted_class, confidence_score, all_probs
    except Exception as e:
        raise Exception(f"Error dalam prediksi: {str(e)}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_image>")
        print("Example: python inference.py dataset/Hawar/001.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: File tidak ditemukan: {image_path}")
        sys.exit(1)
    
    print("Loading model...")
    model = load_model()
    print("Model loaded successfully!")
    
    print(f"\nPredicting image: {image_path}")
    predicted_class, confidence, all_probs = predict_image(image_path, model)
    
    print("\n" + "=" * 50)
    print("HASIL PREDIKSI")
    print("=" * 50)
    print(f"Prediksi: {predicted_class.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print("\nProbabilitas semua kelas:")
    for cls, prob in all_probs.items():
        marker = "✓" if cls == predicted_class else " "
        print(f"  {marker} {cls}: {prob:.2f}%")
    print("=" * 50)

