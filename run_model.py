#!/usr/bin/env python3
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
        print("\nAmbos frameworks están disponibles.")
        choice = input("¿Cuál prefieres usar? (tf/pytorch): ").lower()
        
        if choice in ['tf', 'tensorflow']:
            print("\nUsando TensorFlow...")
            os.system("python minecraft_cnn_detector.py")
        elif choice in ['pt', 'pytorch', 'torch']:
            print("\nUsando PyTorch...")
            os.system("python minecraft_cnn_pytorch.py")
        else:
            print("\nOpcion no valida. Usando TensorFlow por defecto.")
            os.system("python minecraft_cnn_detector.py")
    
    elif tf_available:
        print("\nTensorFlow disponible. Usando TensorFlow...")
        os.system("python minecraft_cnn_detector.py")
    
    elif pt_available:
        print("\nPyTorch disponible. Usando PyTorch...")
        os.system("python minecraft_cnn_pytorch.py")
    
    else:
        print("\nNingun framework de deep learning está disponible.")
        print("Instala TensorFlow o PyTorch:")
        print("   pip install tensorflow-cpu")
        print("   pip install torch torchvision")
        print("\nUsando modelo Random Forest...")
        os.system("python minecraft_block_detector.py")

if __name__ == "__main__":
    main()
