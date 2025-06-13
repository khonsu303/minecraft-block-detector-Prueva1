import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Constants
IMG_SIZE = 64
BATCH_SIZE = 32

def extract_features(img):
    """
    Extrae características importantes de una imagen:
    1. Gradientes (bordes y formas)
    2. Colores (histograma HSV)
    """
    # Redimensiona la imagen a 84x84 píxeles
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Convierte a escala de grises para analizar formas
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calcula gradientes (bordes) en X e Y
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy)
    
    # Divide los ángulos en 9 bins (contenedores)
    bins = np.int32(9 * ang / (2 * np.pi))
    
    # Calcula el histograma de gradientes
    hist = np.zeros(9)
    for i in range(9):
        hist[i] = np.sum(mag[bins == i])
    
    # Normaliza el histograma
    hist = hist / np.sum(hist)
    
    # Añade características de color
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    color_hist = cv2.normalize(color_hist, color_hist).flatten()
    
    # Combina características de gradientes y color
    features = np.concatenate([hist, color_hist])
    
    return features

def load_dataset(base_dir):
    """
    Carga todas las imágenes del dataset:
    1. Lee las carpetas de bloques
    2. Procesa cada imagen
    3. Extrae características
    """
    images = []
    labels = []
    class_names = []
    
    # Obtiene todas las carpetas de bloques
    block_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    block_dirs = block_dirs[:20]  # Use first 20 folders(Estos son los numeros de archivos que se van a sumar )
    print(f"Using these 20 block types: {block_dirs}")
    
    # Procesa cada carpeta de bloques
    for class_idx, block_dir in enumerate(block_dirs):
        class_names.append(block_dir)
        block_path = os.path.join(base_dir, block_dir)
        
        # Procesa cada imagen en la carpeta
        for img_name in os.listdir(block_path):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(block_path, img_name)
                try:
                    # Lee y procesa la imagen
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Could not read image: {img_path}")
                        continue
                    
                    # Extrae características
                    features = extract_features(img)
                    images.append(features)
                    labels.append(class_idx)
                    
                    # Muestra progreso cada 100 imágenes
                    if len(images) % 100 == 0:
                        print(f"Processed {len(images)} images...")
                        
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    
    return np.array(images), np.array(labels), class_names

def train_model():
    """
    Entrena el modelo de clasificación:
    1. Carga los datos
    2. Divide en entrenamiento y prueba
    3. Entrena el modelo
    4. Evalúa el rendimiento
    """
    # Carga el dataset
    print("Loading dataset...")
    X, y, class_names = load_dataset('.')
    
    # Divide los datos (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Normaliza los datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Crea y entrena el modelo
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=300,  # Increased from 200 to 300 for more classes (numero de arboles disponibles 100 por cada 10 bloques)
        max_depth=15,      # Increased from 10 to 15 for more complex patterns (profundidad del arbol 5 POR CADA 10 bloques)
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1            # Usa todos los núcleos de CPU
    )
    model.fit(X_train, y_train)
    
    # Evalúa el modelo
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.2f}")
    print(f"Testing accuracy: {test_score:.2f}")
    
    # Muestra las características más importantes
    feature_importance = model.feature_importances_
    print("\nTop 5 most important features:")
    top_indices = np.argsort(feature_importance)[-5:]
    for idx in top_indices:
        print(f"Feature {idx}: {feature_importance[idx]:.4f}")
    
    # Guarda el modelo y el escalador
    joblib.dump(model, 'minecraft_block_detector.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("\nModel saved as 'minecraft_block_detector.joblib'")
    
    # Guarda los nombres de las clases
    with open('class_names.txt', 'w') as f:
        for name in class_names:
            f.write(name + '\n')
    
    return model, scaler, class_names

def predict_block(model, scaler, class_names, image_path):
    """
    Predice el tipo de bloque en una imagen:
    1. Procesa la imagen
    2. Extrae características
    3. Hace la predicción
    4. Retorna las 3 mejores opciones
    """
    # Lee la imagen
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Extrae características
    features = extract_features(img)
    features = features.reshape(1, -1)
    
    # Normaliza las características
    features = scaler.transform(features)
    
    # Hace la predicción
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)
    confidence = probabilities[0][prediction[0]]
    
    # Obtiene las 3 mejores predicciones
    top_3_idx = np.argsort(probabilities[0])[-3:][::-1]
    top_3_predictions = [(class_names[idx], probabilities[0][idx]) for idx in top_3_idx]
    
    return class_names[prediction[0]], confidence, top_3_predictions

if __name__ == "__main__":
    # Entrena el modelo
    model, scaler, class_names = train_model()
    
    # Prueba con una imagen
    print("\nExample prediction:")
    test_image = "test_image.jpg"
    if os.path.exists(test_image):
        predicted_block, confidence, top_3 = predict_block(model, scaler, class_names, test_image)
        print(f"Predicted block: {predicted_block}")
        print(f"Confidence: {confidence:.2f}")
        print("\nTop 3 predictions:")
        for block, conf in top_3:
            print(f"{block}: {conf:.2f}") 