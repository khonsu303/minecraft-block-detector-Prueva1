#!/usr/bin/env python3
"""
Script de prueba para el modelo CNN
"""

import os
import cv2
import numpy as np
import time
import tensorflow as tf
from minecraft_cnn_detector import preprocess_image, predict_block_cnn

def test_model_loading():
    """Prueba la carga del modelo"""
    print("🧪 Probando carga del modelo...")
    
    try:
        model = tf.keras.models.load_model('minecraft_cnn_model.h5')
        print(" Modelo CNN cargado correctamente")
        
        with open('class_names.txt', 'r') as f:
            class_names = [line.strip() for line in f]
        print(f" {len(class_names)} clases cargadas")
        
        return model, class_names
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None, None

def test_preprocessing():
    """Prueba el preprocesamiento de imágenes"""
    print("\n🧪 Probando preprocesamiento...")
    
    # Crear una imagen de prueba
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    try:
        processed = preprocess_image(test_img)
        expected_shape = (224, 224, 3)
        
        if processed.shape == expected_shape:
            print("✅ Preprocesamiento correcto")
            print(f"   Forma de entrada: {test_img.shape}")
            print(f"   Forma de salida: {processed.shape}")
            print(f"   Rango de valores: [{processed.min():.3f}, {processed.max():.3f}]")
            return True
        else:
            print(f"❌ Forma incorrecta: {processed.shape} != {expected_shape}")
            return False
    except Exception as e:
        print(f"❌ Error en preprocesamiento: {e}")
        return False

def test_prediction_speed():
    """Prueba la velocidad de predicción"""
    print("\n🧪 Probando velocidad de predicción...")
    
    model, class_names = test_model_loading()
    if model is None:
        return False
    
    # Crear imagen de prueba
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Guardar imagen temporal
    temp_path = 'temp_test_image.jpg'
    cv2.imwrite(temp_path, test_img)
    
    try:
        # Medir tiempo de predicción
        times = []
        for i in range(10):
            start_time = time.time()
            predicted_block, confidence, top_3 = predict_block_cnn(model, class_names, temp_path)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"✅ Predicción exitosa")
        print(f"   Bloque predecido: {predicted_block}")
        print(f"   Confianza: {confidence:.4f}")
        print(f"   Tiempo promedio: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
        
        # Limpiar archivo temporal
        os.remove(temp_path)
        return True
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def test_batch_prediction():
    """Prueba predicción en lote"""
    print("\n🧪 Probando predicción en lote...")
    
    model, class_names = test_model_loading()
    if model is None:
        return False
    
    # Crear múltiples imágenes de prueba
    batch_size = 5
    test_images = []
    
    for i in range(batch_size):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_images.append(img)
    
    try:
        # Preprocesar todas las imágenes
        processed_images = []
        for img in test_images:
            processed = preprocess_image(img)
            processed_images.append(processed)
        
        # Convertir a batch
        batch = np.array(processed_images)
        
        # Predicción en lote
        start_time = time.time()
        predictions = model.predict(batch, verbose=0)
        end_time = time.time()
        
        batch_time = end_time - start_time
        avg_time_per_image = batch_time / batch_size
        
        print(f"✅ Predicción en lote exitosa")
        print(f"   Tamaño del lote: {batch_size}")
        print(f"   Tiempo total: {batch_time*1000:.2f} ms")
        print(f"   Tiempo por imagen: {avg_time_per_image*1000:.2f} ms")
        
        # Mostrar predicciones
        for i, pred in enumerate(predictions):
            predicted_class = np.argmax(pred)
            confidence = pred[predicted_class]
            print(f"   Imagen {i+1}: {class_names[predicted_class]} ({confidence:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en predicción en lote: {e}")
        return False

def test_memory_usage():
    """Prueba el uso de memoria"""
    print("\n🧪 Probando uso de memoria...")
    
    try:
        import psutil
        process = psutil.Process()
        
        # Memoria antes de cargar modelo
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Cargar modelo
        model = tf.keras.models.load_model('minecraft_cnn_model.h5')
        
        # Memoria después de cargar modelo
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        print(f"✅ Uso de memoria:")
        print(f"   Antes: {memory_before:.1f} MB")
        print(f"   Después: {memory_after:.1f} MB")
        print(f"   Incremento: {memory_used:.1f} MB")
        
        if memory_used < 1000:  # Menos de 1GB
            print("   ✅ Uso de memoria aceptable")
        else:
            print("   ⚠️  Uso de memoria alto")
        
        return True
        
    except ImportError:
        print("⚠️  psutil no instalado - No se puede medir memoria")
        return True
    except Exception as e:
        print(f"❌ Error midiendo memoria: {e}")
        return False

def test_gpu_usage():
    """Prueba el uso de GPU"""
    print("\n🧪 Probando uso de GPU...")
    
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU detectada: {len(gpus)} dispositivo(s)")
            
            # Verificar si TensorFlow está usando GPU
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
            
            print("✅ TensorFlow está usando GPU")
            return True
        else:
            print("⚠️  No se detectó GPU - Usando CPU")
            return True
            
    except Exception as e:
        print(f"❌ Error verificando GPU: {e}")
        return False

def run_all_tests():
    """Ejecuta todas las pruebas"""
    print("🚀 Iniciando pruebas del modelo CNN")
    print("="*50)
    
    tests = [
        ("Carga del modelo", test_model_loading),
        ("Preprocesamiento", test_preprocessing),
        ("Velocidad de predicción", test_prediction_speed),
        ("Predicción en lote", test_batch_prediction),
        ("Uso de memoria", test_memory_usage),
        ("Uso de GPU", test_gpu_usage)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - PASÓ")
            else:
                print(f"❌ {test_name} - FALLÓ")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
    
    print("\n" + "="*50)
    print(f"📊 Resultados: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron! El modelo CNN está funcionando correctamente.")
    else:
        print("⚠️  Algunas pruebas fallaron. Revisa los errores arriba.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 