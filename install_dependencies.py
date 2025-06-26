#!/usr/bin/env python3
"""
Script para instalar dependencias de manera robusta
"""

import sys
import subprocess
import platform

def check_python_version():
    """Verifica la versión de Python"""
    print("🔍 Verificando Python...")
    print(f"   Versión: {sys.version}")
    
    if sys.version_info >= (3, 8):
        print("✅ Python compatible")
        return True
    else:
        print("❌ Python 3.8+ requerido")
        return False

def install_basic_dependencies():
    """Instala dependencias básicas"""
    print("\n📦 Instalando dependencias básicas...")
    
    basic_deps = [
        "flask>=2.3.0",
        "flask-cors>=4.0.0", 
        "opencv-python>=4.8.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "Werkzeug>=3.0.0",
        "matplotlib>=3.7.0",
        "pillow>=10.0.0"
    ]
    
    for dep in basic_deps:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error instalando {dep}: {e}")
            return False
    
    return True

def try_install_tensorflow():
    """Intenta instalar TensorFlow"""
    print("\n📦 Intentando instalar TensorFlow...")
    
    tf_versions = [
        "tensorflow-cpu>=2.15.0",
        "tensorflow-cpu==2.14.0",
        "tensorflow-cpu==2.13.0",
        "tensorflow>=2.15.0",
        "tensorflow==2.14.0"
    ]
    
    for version in tf_versions:
        try:
            print(f"   Probando: {version}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", version])
            print(f"✅ TensorFlow instalado: {version}")
            return True
        except subprocess.CalledProcessError:
            print(f"   ❌ {version} falló")
            continue
    
    return False

def try_install_pytorch():
    """Intenta instalar PyTorch"""
    print("\n📦 Intentando instalar PyTorch...")
    
    try:
        # Instalar PyTorch CPU (más compatible)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"])
        print("✅ PyTorch instalado correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando PyTorch: {e}")
        return False

def test_tensorflow():
    """Prueba TensorFlow"""
    print("\n🧪 Probando TensorFlow...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} funciona")
        
        # Probar operación básica
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.add(a, b)
        print(f"✅ Operación básica: {c.numpy()}")
        
        return True
    except ImportError as e:
        print(f"❌ TensorFlow no disponible: {e}")
        return False
    except Exception as e:
        print(f"❌ Error probando TensorFlow: {e}")
        return False

def test_pytorch():
    """Prueba PyTorch"""
    print("\n🧪 Probando PyTorch...")
    
    try:
        import torch
        import torchvision
        print(f"✅ PyTorch {torch.__version__} funciona")
        print(f"✅ TorchVision {torchvision.__version__} funciona")
        
        # Probar operación básica
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        c = a + b
        print(f"✅ Operación básica: {c.numpy()}")
        
        # Verificar dispositivo
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ Dispositivo disponible: {device}")
        
        return True
    except ImportError as e:
        print(f"❌ PyTorch no disponible: {e}")
        return False
    except Exception as e:
        print(f"❌ Error probando PyTorch: {e}")
        return False

def create_model_selection():
    """Crea un script para seleccionar el modelo a usar"""
    print("\n📝 Creando selector de modelo...")
    
    selector_content = '''#!/usr/bin/env python3
"""
Selector de modelo para Minecraft Block Detector
"""

import os
import sys

def check_tensorflow():
    """Verifica si TensorFlow está disponible"""
    try:
        import tensorflow as tf
        return True
    except ImportError:
        return False

def check_pytorch():
    """Verifica si PyTorch está disponible"""
    try:
        import torch
        import torchvision
        return True
    except ImportError:
        return False

def main():
    """Función principal"""
    print("Selector de Modelo - Minecraft Block Detector")
    print("="*50)
    
    tf_available = check_tensorflow()
    pt_available = check_pytorch()
    
    print(f"TensorFlow disponible: {'SI' if tf_available else 'NO'}")
    print(f"PyTorch disponible: {'SI' if pt_available else 'NO'}")
    
    if tf_available and pt_available:
        print("\\nAmbos frameworks están disponibles.")
        choice = input("¿Cuál prefieres usar? (tf/pytorch): ").lower()
        
        if choice in ['tf', 'tensorflow']:
            print("\\nUsando TensorFlow...")
            os.system("python minecraft_cnn_detector.py")
        elif choice in ['pt', 'pytorch', 'torch']:
            print("\\nUsando PyTorch...")
            os.system("python minecraft_cnn_pytorch.py")
        else:
            print("\\nOpcion no valida. Usando TensorFlow por defecto.")
            os.system("python minecraft_cnn_detector.py")
    
    elif tf_available:
        print("\\nTensorFlow disponible. Usando TensorFlow...")
        os.system("python minecraft_cnn_detector.py")
    
    elif pt_available:
        print("\\nPyTorch disponible. Usando PyTorch...")
        os.system("python minecraft_cnn_pytorch.py")
    
    else:
        print("\\nNingun framework de deep learning está disponible.")
        print("Instala TensorFlow o PyTorch:")
        print("   pip install tensorflow-cpu")
        print("   pip install torch torchvision")
        print("\\nUsando modelo Random Forest...")
        os.system("python minecraft_block_detector.py")

if __name__ == "__main__":
    main()
'''
    
    with open('run_model.py', 'w', encoding='utf-8') as f:
        f.write(selector_content)
    
    print("✅ Selector creado: run_model.py")

def main():
    """Función principal"""
    print("🚀 Instalador de Dependencias - Minecraft Block Detector")
    print("="*60)
    
    # Verificar Python
    if not check_python_version():
        return
    
    # Instalar dependencias básicas
    if not install_basic_dependencies():
        print("❌ Error instalando dependencias básicas")
        return
    
    # Intentar instalar frameworks de deep learning
    tf_success = try_install_tensorflow()
    pt_success = try_install_pytorch()
    
    # Probar frameworks
    tf_works = test_tensorflow() if tf_success else False
    pt_works = test_pytorch() if pt_success else False
    
    # Crear selector de modelo
    create_model_selection()
    
    # Resumen
    print("\n" + "="*50)
    print("📊 Resumen de Instalación")
    print("="*50)
    print(f"TensorFlow: {'✅' if tf_works else '❌'}")
    print(f"PyTorch: {'✅' if pt_works else '❌'}")
    
    if tf_works or pt_works:
        print("\n🎉 ¡Instalación completada!")
        print("\n📋 Próximos pasos:")
        print("1. Ejecutar: python run_model.py")
        print("2. O directamente:")
        if tf_works:
            print("   python minecraft_cnn_detector.py")
        if pt_works:
            print("   python minecraft_cnn_pytorch.py")
        print("3. Probar: python test_cnn.py")
    else:
        print("\n⚠️ Ningún framework de deep learning funciona")
        print("💡 Usando modelo Random Forest como fallback")
        print("   python minecraft_block_detector.py")

if __name__ == "__main__":
    main() 