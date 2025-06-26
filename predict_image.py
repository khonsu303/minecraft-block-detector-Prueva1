import torch
import os
from minecraft_cnn_pytorch import MinecraftCNN, predict_block_pytorch

# Cargar nombres de clases
def load_class_names(filename='class_names.txt'):
    with open(filename, 'r') as f:
        return [line.strip() for line in f]

class_names = load_class_names()
num_classes = len(class_names)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inicializar el modelo y cargar pesos
model = MinecraftCNN(num_classes).to(device)
model.load_state_dict(torch.load('minecraft_pytorch_model.pth', map_location=device))
model.eval()

# Imagen a predecir
image_path = 'test_image.jpg'  # Cambia el nombre si tu imagen es otra

if os.path.exists(image_path):
    predicted_block, confidence, top_3 = predict_block_pytorch(model, class_names, image_path)
    print(f"El bloque predecido es: {predicted_block}")
    print(f"Confianza: {confidence:.4f}")
    print("\nTop 3 predicciones:")
    for block, conf in top_3:
        print(f"{block}: {conf:.4f}")
else:
    print(f"No se encontró la imagen: {image_path}") 