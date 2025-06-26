import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib

# Configuración para evitar problemas de memoria en GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Constants
IMG_SIZE = 224  # Tamaño estándar para modelos pre-entrenados
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

def create_cnn_model(num_classes):
    """
    Crea un modelo CNN basado en MobileNetV2 con transfer learning
    """
    # Modelo base pre-entrenado
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Congela las capas del modelo base
    base_model.trainable = False
    
    # Construye el modelo completo
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compila el modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_image(img):
    """
    Preprocesa una imagen para el modelo CNN
    """
    # Redimensiona la imagen
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Convierte de BGR a RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normaliza los valores de píxeles
    img = img.astype(np.float32) / 255.0
    
    return img

def load_dataset_cnn(base_dir, progress_callback=None):
    """
    Carga el dataset para entrenamiento CNN
    """
    images = []
    labels = []
    class_names = []
    
    # Obtiene todas las carpetas de bloques
    block_dirs = [d for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d)) 
                  and not d.startswith('.') 
                  and not d.startswith('__') 
                  and d not in ['venv', 'node_modules', 'build', 'public', 'src', 'uploads']]
    
    # Usar todas las carpetas de bloques disponibles
    # (Ya no se limita a 15 clases)
    
    if progress_callback:
        progress_callback(f"Usando {len(block_dirs)} tipos de bloques: {block_dirs}")
    else:
        print(f"Usando {len(block_dirs)} tipos de bloques: {block_dirs}")
    
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
                            progress_callback(f"No se pudo leer la imagen: {img_path}")
                        continue
                    
                    # Preprocesa la imagen
                    processed_img = preprocess_image(img)
                    images.append(processed_img)
                    labels.append(class_idx)
                    
                    # Muestra progreso cada 100 imágenes
                    if len(images) % 100 == 0:
                        msg = f"Procesadas {len(images)} imágenes..."
                        if progress_callback:
                            progress_callback(msg)
                        else:
                            print(msg)
                        
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"Error cargando imagen {img_path}: {e}")
    
    return np.array(images), np.array(labels), class_names

def create_data_generators(X_train, y_train, X_val, y_val):
    """
    Crea generadores de datos con data augmentation
    """
    # Data augmentation para entrenamiento
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Solo normalización para validación
    val_datagen = ImageDataGenerator()
    
    # Generadores
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    return train_generator, val_generator

def train_cnn_model(progress_callback=None):
    """
    Entrena el modelo CNN
    """
    # Carga el dataset
    if progress_callback:
        progress_callback("Cargando dataset...")
    else:
        print("Cargando dataset...")
    
    X, y, class_names = load_dataset_cnn('.', progress_callback=progress_callback)
    
    if progress_callback:
        progress_callback(f"Dataset cargado: {len(X)} imágenes, {len(class_names)} clases")
    else:
        print(f"Dataset cargado: {len(X)} imágenes, {len(class_names)} clases")
    
    # Divide los datos
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Crea generadores de datos
    train_generator, val_generator = create_data_generators(X_train, y_train, X_val, y_val)
    
    # Crea el modelo
    if progress_callback:
        progress_callback("Creando modelo CNN...")
    else:
        print("Creando modelo CNN...")
    
    model = create_cnn_model(len(class_names))
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
    ]
    
    # Entrena el modelo
    if progress_callback:
        progress_callback("Entrenando modelo...")
    else:
        print("Entrenando modelo...")
    
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=len(X_val) // BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evalúa el modelo
    val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
    
    if progress_callback:
        progress_callback(f"Precisión de validación: {val_accuracy:.4f}")
        progress_callback(f"Pérdida de validación: {val_loss:.4f}")
    else:
        print(f"Precisión de validación: {val_accuracy:.4f}")
        print(f"Pérdida de validación: {val_loss:.4f}")
    
    # Guarda el modelo
    model.save('minecraft_cnn_model.h5')
    
    # Guarda los nombres de las clases
    with open('class_names.txt', 'w') as f:
        for name in class_names:
            f.write(name + '\n')
    
    if progress_callback:
        progress_callback("Modelo guardado como 'minecraft_cnn_model.h5'")
        progress_callback("Entrenamiento finalizado")
    else:
        print("Modelo guardado como 'minecraft_cnn_model.h5'")
    
    return model, class_names, history

def predict_block_cnn(model, class_names, image_path):
    """
    Predice el tipo de bloque usando el modelo CNN
    """
    # Lee la imagen
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")
    
    # Preprocesa la imagen
    processed_img = preprocess_image(img)
    processed_img = np.expand_dims(processed_img, axis=0)
    
    # Hace la predicción
    predictions = model.predict(processed_img)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Obtiene las 3 mejores predicciones
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [(class_names[idx], float(predictions[0][idx])) for idx in top_3_idx]
    
    return class_names[predicted_class], confidence, top_3_predictions

def plot_training_history(history):
    """
    Grafica el historial de entrenamiento
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Precisión
    ax1.plot(history.history['accuracy'], label='Entrenamiento')
    ax1.plot(history.history['val_accuracy'], label='Validación')
    ax1.set_title('Precisión del Modelo')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Precisión')
    ax1.legend()
    
    # Pérdida
    ax2.plot(history.history['loss'], label='Entrenamiento')
    ax2.plot(history.history['val_loss'], label='Validación')
    ax2.set_title('Pérdida del Modelo')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Pérdida')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == "__main__":
    # Entrena el modelo
    model, class_names, history = train_cnn_model()
    
    # Grafica el historial de entrenamiento
    plot_training_history(history)
    
    # Prueba con una imagen
    print("\nEjemplo de predicción:")
    test_image = "test_image.jpg"
    if os.path.exists(test_image):
        predicted_block, confidence, top_3 = predict_block_cnn(model, class_names, test_image)
        print(f"El bloque predecido es: {predicted_block}")
        print(f"Confianza: {confidence:.4f}")
        print("\nTop 3 predicciones:")
        for block, conf in top_3:
            print(f"{block}: {conf:.4f}") 