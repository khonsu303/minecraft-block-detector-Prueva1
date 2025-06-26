import tensorflow as tf

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU detectada:")
    for gpu in gpus:
        print("  -", gpu)
else:
    print("❌ No se detectó GPU. El entrenamiento será en CPU.") 