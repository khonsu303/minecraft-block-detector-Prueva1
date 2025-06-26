#!/usr/bin/env python3
"""
Script de configuración para migrar a CNN
"""

import os
import sys
import subprocess
import shutil
import torch

def check_python_version():
    """Verifica la versión de Python"""
    if sys.version_info < (3, 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} - OK")
    return True

def install_requirements():
    """Instala las dependencias"""
    print("\n📦 Instalando dependencias...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencias instaladas correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def check_gpu():
    """Verifica si hay GPU disponible"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU detectada: {len(gpus)} dispositivo(s)")
            for gpu in gpus:
                print(f"   - {gpu.name}")
            return True
        else:
            print("⚠️  No se detectó GPU - El entrenamiento será más lento")
            return False
    except ImportError:
        print("⚠️  TensorFlow no instalado - No se puede verificar GPU")
        return False

def backup_old_model():
    """Hace backup del modelo anterior"""
    old_files = ['minecraft_block_detector.joblib', 'scaler.joblib']
    backup_dir = 'backup_old_model'
    
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    for file in old_files:
        if os.path.exists(file):
            shutil.copy2(file, backup_dir)
            print(f"✅ Backup creado: {file} → {backup_dir}/")
    
    if os.path.exists(backup_dir):
        print(f"📁 Backup completo en: {backup_dir}/")

def create_directories():
    """Crea directorios necesarios"""
    dirs = ['uploads', 'models', 'logs']
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"📁 Directorio creado: {dir_name}/")

def test_imports():
    """Prueba las importaciones principales"""
    print("\n🧪 Probando importaciones...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
        print(tf.config.list_physical_devices('GPU'))
    except ImportError as e:
        print(f"❌ Error importando TensorFlow: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"❌ Error importando OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ Error importando NumPy: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Error importando Matplotlib: {e}")
        return False
    
    return True

def create_config_file():
    """Crea archivo de configuración"""
    config_content = """# Configuración del Modelo CNN
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Configuración de Data Augmentation
ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
ZOOM_RANGE = 0.2
BRIGHTNESS_RANGE = [0.8, 1.2]

# Configuración del Servidor
FLASK_PORT = 5000
FLASK_DEBUG = True
"""
    
    with open('config.py', 'w') as f:
        f.write(config_content)
    print("✅ Archivo de configuración creado: config.py")

def show_next_steps():
    """Muestra los próximos pasos"""
    print("\n" + "="*50)
    print("🎉 ¡Configuración completada!")
    print("="*50)
    print("\n📋 Próximos pasos:")
    print("1. Entrenar el modelo CNN:")
    print("   python minecraft_cnn_detector.py")
    print("\n2. Comparar con el modelo anterior:")
    print("   python compare_models.py")
    print("\n3. Iniciar el servidor web:")
    print("   python app.py")
    print("\n4. (Opcional) Probar el frontend:")
    print("   npm install && npm start")
    print("\n📚 Documentación:")
    print("   - README.md: Guía completa")
    print("   - config.py: Configuración del modelo")
    print("\n🆘 Si tienes problemas:")
    print("   - Verifica que TensorFlow esté instalado correctamente")
    print("   - Asegúrate de tener suficiente memoria RAM (4GB+)")
    print("   - Para GPU: instala CUDA y cuDNN")

def main():
    """Función principal"""
    print("🚀 Configurando Minecraft Block Detector - CNN Edition")
    print("="*60)
    
    # Verificaciones iniciales
    if not check_python_version():
        sys.exit(1)
    
    # Instalar dependencias
    if not install_requirements():
        print("❌ Error en la instalación. Revisa los errores arriba.")
        sys.exit(1)
    
    # Verificar GPU
    check_gpu()
    
    # Backup del modelo anterior
    backup_old_model()
    
    # Crear directorios
    create_directories()
    
    # Probar importaciones
    if not test_imports():
        print("❌ Error en las importaciones. Reinstala las dependencias.")
        sys.exit(1)
    
    # Crear archivo de configuración
    create_config_file()
    
    # Mostrar próximos pasos
    show_next_steps()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

if __name__ == "__main__":
    main() 