# Minecraft Block Detector

Este proyecto es un sistema de detección de bloques de Minecraft utilizando visión por computadora y aprendizaje automático. El sistema puede identificar diferentes tipos de bloques de Minecraft a partir de imágenes.

## Características

- Detección de múltiples tipos de bloques de Minecraft
- Clasificación con Random Forest
- Extracción de características usando OpenCV
- Predicción de los 3 bloques más probables con sus niveles de confianza

## Requisitos

- Python 3.8+
- OpenCV
- scikit-learn
- numpy
- joblib

## Instalación

1. Clona este repositorio:
```bash
git clone https://github.com/tu-usuario/minecraft-block-detector.git
cd minecraft-block-detector
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

1. Coloca tus imágenes de bloques en carpetas separadas por tipo de bloque
2. Ejecuta el script principal:
```bash
python minecraft_block_detector.py
```

3. Para probar una imagen específica:
```bash
python minecraft_block_detector.py --test test_image.jpg
```

## Estructura del Proyecto

- `minecraft_block_detector.py`: Script principal
- `requirements.txt`: Dependencias del proyecto
- `/bloques/`: Directorio con las imágenes de entrenamiento organizadas por tipo de bloque

## Notas

- El modelo está entrenado con 20 tipos diferentes de bloques
- Se utilizan 300 árboles en el Random Forest (100 por cada 10 bloques)
- La profundidad máxima del árbol es 15 (5 por cada 10 bloques)

## Licencia

Este proyecto está bajo la Licencia MIT. 