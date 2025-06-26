import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")


IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

class MinecraftBlockDataset(Dataset):
    """Dataset personalizado para bloques de Minecraft"""
    
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class MinecraftCNN(nn.Module):
    """Modelo CNN para clasificación de bloques de Minecraft"""
    
    def __init__(self, num_classes):
        super(MinecraftCNN, self).__init__()
        
        self.backbone = models.resnet18(pretrained=True)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def preprocess_image(img):
    """Preprocesa una imagen para el modelo PyTorch"""
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1) 
    img = img / 255.0
    
    return img

def load_dataset_pytorch(base_dir, progress_callback=None):
    """Carga el dataset para entrenamiento PyTorch"""
    images = []
    labels = []
    class_names = []
    
    block_dirs = [d for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d)) 
                  and not d.startswith('.') 
                  and not d.startswith('__') 
                  and d not in ['venv', 'node_modules', 'build', 'public', 'src', 'uploads']]
    

    block_dirs = block_dirs[:25]

    
    if progress_callback:
        progress_callback(f"Usando {len(block_dirs)} tipos de bloques: {block_dirs}")
    else:
        print(f"Usando {len(block_dirs)} tipos de bloques: {block_dirs}")
    
    for class_idx, block_dir in enumerate(block_dirs):
        class_names.append(block_dir)
        block_path = os.path.join(base_dir, block_dir)
        
        for img_name in os.listdir(block_path):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(block_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        if progress_callback:
                            progress_callback(f"No se pudo leer la imagen: {img_path}")
                        continue
                    
                    processed_img = preprocess_image(img)
                    images.append(processed_img)
                    labels.append(class_idx)
                    
                    if len(images) % 100 == 0:
                        msg = f"Procesadas {len(images)} imágenes..."
                        if progress_callback:
                            progress_callback(msg)
                        else:
                            print(msg)
                        
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"Error cargando imagen {img_path}: {e}")
    
    return images, labels, class_names

def create_transforms():
    """Crea transformaciones para data augmentation"""
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_pytorch_model(progress_callback=None):
    """Entrena el modelo PyTorch"""
    if progress_callback:
        progress_callback("Cargando dataset...")
    else:
        print("Cargando dataset...")
    
    images, labels, class_names = load_dataset_pytorch('.', progress_callback=progress_callback)
    
    if progress_callback:
        progress_callback(f"Dataset cargado: {len(images)} imágenes, {len(class_names)} clases")
    else:
        print(f"Dataset cargado: {len(images)} imágenes, {len(class_names)} clases")
    
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_transform, val_transform = create_transforms()
    
    # Crea datasets
    train_dataset = MinecraftBlockDataset(X_train, y_train, transform=train_transform)
    val_dataset = MinecraftBlockDataset(X_val, y_val, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    if progress_callback:
        progress_callback("Creando modelo PyTorch...")
    else:
        print("Creando modelo PyTorch...")
    
    model = MinecraftCNN(len(class_names)).to(device)
    
    # Criterio y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    if progress_callback:
        progress_callback("Entrenando modelo...")
    else:
        print("Entrenando modelo...")
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                msg = f"Época {epoch+1}/{EPOCHS}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
                if progress_callback:
                    progress_callback(msg)
                else:
                    print(msg)
        
        # Validación
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        scheduler.step(val_loss)
        
        msg = f"Época {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'minecraft_pytorch_model.pth')
    
    with open('class_names.txt', 'w') as f:
        for name in class_names:
            f.write(name + '\n')
    
    if progress_callback:
        progress_callback(f"Mejor precisión de validación: {best_val_acc:.2f}%")
        progress_callback("Modelo guardado como 'minecraft_pytorch_model.pth'")
        progress_callback("Entrenamiento finalizado")
    else:
        print(f"Mejor precisión de validación: {best_val_acc:.2f}%")
        print("Modelo guardado como 'minecraft_pytorch_model.pth'")
    
    return model, class_names, (train_losses, val_losses, val_accuracies)

def predict_block_pytorch(model, class_names, image_path):
    """Predice el tipo de bloque usando el modelo PyTorch"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")
    
    processed_img = preprocess_image(img)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    processed_img = normalize(processed_img)
    
    processed_img = processed_img.unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(processed_img)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    top_3_idx = torch.argsort(probabilities[0], descending=True)[:3]
    top_3_predictions = [(class_names[idx], float(probabilities[0][idx])) for idx in top_3_idx]
    
    return class_names[predicted_class], confidence, top_3_predictions

def plot_training_history(history):
    """Grafica el historial de entrenamiento"""
    train_losses, val_losses, val_accuracies = history
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
  
    ax1.plot(train_losses, label='Entrenamiento')
    ax1.plot(val_losses, label='Validación')
    ax1.set_title('Pérdida del Modelo')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.legend()
    
    ax2.plot(val_accuracies, label='Validación')
    ax2.set_title('Precisión del Modelo')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Precisión (%)')
    ax2.legend()
    
    plt.tight_layout()
    if not os.path.exists('static'):
        os.makedirs('static')
    plt.savefig('static/training_history_pytorch.png')
    plt.show()

if __name__ == "__main__":
    model, class_names, history = train_pytorch_model()
    
    plot_training_history(history)
    
    print("\nEjemplo de predicción:")
    test_image = "test_image.jpg"
    if os.path.exists(test_image):
        predicted_block, confidence, top_3 = predict_block_pytorch(model, class_names, test_image)
        print(f"El bloque predecido es: {predicted_block}")
        print(f"Confianza: {confidence:.4f}")
        print("\nTop 3 predicciones:")
        for block, conf in top_3:
            print(f"{block}: {conf:.4f}") 