#!/usr/bin/env python3
"""
Script para instalar TensorFlow correctamente en Windows
"""

import sys
import subprocess
import platform

def check_system():
    """Verifica el sistema operativo y versión de Python"""
    print("🔍 Verificando sistema...")
    print(f"   Sistema: {platform.system()} {platform.release()}")
    print(f"   Arquitectura: {platform.architecture()[0]}")
    print(f"   Python: {sys.version}")
    
    # Verificar versión de Python
    if sys.version_info >= (3, 12):
        print("⚠️  Python 3.12+ detectado - TensorFlow puede tener problemas")
        return False
    elif sys.version_info >= (3, 8):
        print("✅ Versión de Python compatible")
        return True
    else:
        print("❌ Python 3.8+ requerido")
        return False

def install_tensorflow_cpu():
    """Instala TensorFlow CPU"""
    print("\n📦 Instalando TensorFlow CPU...")
    
    try:
        # Primero actualizar pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Instalar TensorFlow CPU
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow-cpu>=2.15.0"])
        print("✅ TensorFlow CPU instalado correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando TensorFlow CPU: {e}")
        return False

def install_tensorflow_gpu():
    """Intenta instalar TensorFlow GPU"""
    print("\n📦 Intentando instalar TensorFlow GPU...")
    
    try:
        # Verificar si hay CUDA disponible
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow>=2.15.0"])
        print("✅ TensorFlow GPU instalado correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando TensorFlow GPU: {e}")
        return False

def install_alternative_tensorflow():
    """Instala una versión alternativa de TensorFlow"""
    print("\n📦 Intentando versión alternativa de TensorFlow...")
    
    alternatives = [
        "tensorflow-cpu==2.14.0",
        "tensorflow-cpu==2.13.0",
        "tensorflow-cpu==2.12.0",
        "tensorflow==2.14.0",
        "tensorflow==2.13.0"
    ]
    
    for version in alternatives:
        try:
            print(f"   Probando: {version}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", version])
            print(f"✅ {version} instalado correctamente")
            return True
        except subprocess.CalledProcessError:
            print(f"   ❌ {version} falló")
            continue
    
    return False

def install_other_dependencies():
    """Instala otras dependencias"""
    print("\n📦 Instalando otras dependencias...")
    
    dependencies = [
        "matplotlib>=3.7.0",
        "pillow>=10.0.0",
        "psutil>=5.9.0"
    ]
    
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep} instalado")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error instalando {dep}: {e}")

def test_tensorflow():
    """Prueba la instalación de TensorFlow"""
    print("\n🧪 Probando TensorFlow...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} importado correctamente")
        
        # Probar operación básica
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.add(a, b)
        print(f"✅ Operación básica exitosa: {c.numpy()}")
        
        # Verificar dispositivos
        devices = tf.config.list_physical_devices()
        print(f"✅ Dispositivos disponibles: {len(devices)}")
        
        return True
    except ImportError as e:
        print(f"❌ Error importando TensorFlow: {e}")
        return False
    except Exception as e:
        print(f"❌ Error probando TensorFlow: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 Instalador de TensorFlow para Windows")
    print("="*50)
    
    # Verificar sistema
    if not check_system():
        print("\n❌ Sistema no compatible")
        return
    
    # Intentar diferentes métodos de instalación
    success = False
    
    # Método 1: TensorFlow CPU
    if not success:
        success = install_tensorflow_cpu()
    
    # Método 2: TensorFlow GPU
    if not success:
        success = install_tensorflow_gpu()
    
    # Método 3: Versiones alternativas
    if not success:
        success = install_alternative_tensorflow()
    
    if not success:
        print("\n❌ No se pudo instalar TensorFlow")
        print("\n💡 Soluciones alternativas:")
        print("1. Usar Python 3.11 o 3.10")
        print("2. Instalar desde conda: conda install tensorflow")
        print("3. Usar Docker con TensorFlow")
        return
    
    # Instalar otras dependencias
    install_other_dependencies()
    
    # Probar instalación
    if test_tensorflow():
        print("\n🎉 ¡TensorFlow instalado correctamente!")
        print("\n📋 Próximos pasos:")
        print("1. Ejecutar: python minecraft_cnn_detector.py")
        print("2. O ejecutar: python test_cnn.py")
    else:
        print("\n❌ TensorFlow no funciona correctamente")

if __name__ == "__main__":
    main() 