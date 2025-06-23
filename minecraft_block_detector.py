import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Constants
IMG_SIZE = 128  # Aumentado de 64 a 128 para capturar más detalles
BATCH_SIZE = 32

def extract_features(img):
    """
    Extrae características mejoradas de una imagen:
    1. Gradientes (bordes y formas)
    2. Colores (histograma HSV)
    3. Textura (GLCM)
    4. Características de forma
    """
    # Redimensiona la imagen
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Convierte a escala de grises para analizar formas
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Características de gradientes mejoradas
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy)
    
    # Histograma de gradientes orientados (HOG)
    bins = np.int32(16 * ang / (2 * np.pi))  # Aumentado de 9 a 16 bins
    hist = np.zeros(16)
    for i in range(16):
        hist[i] = np.sum(mag[bins == i])
    hist = hist / np.sum(hist)
    
    # 2. Características de color mejoradas
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_hist = cv2.calcHist([hsv], [0, 1], None, [12, 12], [0, 180, 0, 256])  # Aumentado de 8x8 a 12x12
    color_hist = cv2.normalize(color_hist, color_hist).flatten()
    
    # 3. Características de textura
    # Calcula la matriz de co-ocurrencia de niveles de gris (GLCM)
    glcm = np.zeros((8, 8), dtype=np.uint8)
    for i in range(IMG_SIZE-1):
        for j in range(IMG_SIZE-1):
            glcm[gray[i,j]//32, gray[i+1,j]//32] += 1
    glcm = glcm.flatten()
    glcm = glcm / np.sum(glcm)
    
    # 4. Características de forma
    # Encuentra contornos
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calcula características de forma
    shape_features = []
    if contours:
        # Área
        area = cv2.contourArea(contours[0])
        # Perímetro
        perimeter = cv2.arcLength(contours[0], True)
        # Circularidad
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        # Rectangularidad
        x, y, w, h = cv2.boundingRect(contours[0])
        rectangularity = area / (w * h) if w * h > 0 else 0
        
        shape_features = [area/(IMG_SIZE*IMG_SIZE), perimeter/(2*IMG_SIZE), circularity, rectangularity]
    else:
        shape_features = [0, 0, 0, 0]
    
    # Combina todas las características
    features = np.concatenate([hist, color_hist, glcm, shape_features])
    
    return features

def load_dataset(base_dir, progress_callback=None):
    """
    Carga todas las imágenes del dataset
    """
    images = []
    labels = []
    class_names = []
    
    # Obtiene todas las carpetas de bloques
    block_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('.') and not d.startswith('__') and d not in ['venv', 'node_modules', 'build', 'public', 'src', 'uploads']]
    block_dirs = block_dirs[:10]  # Limitar a solo 10 tipos de bloques
    if progress_callback:
        progress_callback(f"Using {len(block_dirs)} block types: {block_dirs}")
    else:
        print(f"Using {len(block_dirs)} block types: {block_dirs}")
    
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
                        if progress_callback:
                            progress_callback(f"Could not read image: {img_path}")
                        continue
                    
                    # Extrae características
                    features = extract_features(img)
                    images.append(features)
                    labels.append(class_idx)
                    
                    # Muestra progreso cada 100 imágenes
                    if len(images) % 100 == 0:
                        msg = f"Processed {len(images)} images..."
                        if progress_callback:
                            progress_callback(msg)
                        else:
                            print(msg)
                        
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"Error loading image {img_path}: {e}")
    
    return np.array(images), np.array(labels), class_names

def train_model(progress_callback=None):
    """
    Entrena el modelo de clasificación mejorado
    Si se pasa progress_callback, se llama con mensajes de progreso.
    """
    # Carga el dataset
    if progress_callback:
        progress_callback("Cargando dataset...")
    else:
        print("Loading dataset...")
    X, y, class_names = load_dataset('.', progress_callback=progress_callback)
    
    # Divide los datos (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Normaliza los datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Crea y entrena el modelo mejorado
    if progress_callback:
        progress_callback("Entrenando modelo...")
    else:
        print("Training model...")
    model = RandomForestClassifier(
        n_estimators=500,  # Aumentado para mejor precisión
        max_depth=20,      # Aumentado para capturar más patrones
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,         # Usa todos los núcleos de CPU
        class_weight='balanced'  # Maneja mejor el desbalance de clases
    )
    model.fit(X_train, y_train)
    
    # Evalúa el modelo
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    if progress_callback:
        progress_callback(f"Precisión entrenamiento: {train_score:.2f}")
        progress_callback(f"Precisión prueba: {test_score:.2f}")
    else:
        print(f"Training accuracy: {train_score:.2f}")
        print(f"Testing accuracy: {test_score:.2f}")
    
    # Muestra las características más importantes
    feature_importance = model.feature_importances_
    top_indices = np.argsort(feature_importance)[-5:]
    for idx in top_indices:
        msg = f"Feature {idx}: {feature_importance[idx]:.4f}"
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)
    
    # Guarda el modelo y el escalador
    joblib.dump(model, 'minecraft_block_detector.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    if progress_callback:
        progress_callback("Modelo guardado como 'minecraft_block_detector.joblib'")
    else:
        print("\nModelo guardado como 'minecraft_block_detector.joblib'")
    
    # Guarda los nombres de las clases
    with open('class_names.txt', 'w') as f:
        for name in class_names:
            f.write(name + '\n')
    if progress_callback:
        progress_callback("Entrenamiento finalizado")
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
        raise ValueError(f"No se pudo leer la imagen: {image_path}")
    
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
        print(f"El bloque predecido es: {predicted_block}")
        print(f"coicidencia: {confidence:.2f}")
        print("\nTop 3 predicciones:")
        for block, conf in top_3:
            print(f"{block}: {conf:.2f}") 