from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import torch
from minecraft_cnn_pytorch import MinecraftCNN, predict_block_pytorch
import threading
import queue

app = Flask(__name__, static_folder='build', static_url_path='')
CORS(app)  # Esto permite las peticiones desde el frontend

# Cargar el modelo PyTorch
with open('class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f]
num_classes = len(class_names)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pytorch_model = MinecraftCNN(num_classes).to(device)
pytorch_model.load_state_dict(torch.load('minecraft_pytorch_model.pth', map_location=device))
pytorch_model.eval()

# Configuración para subir archivos
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/blocks', methods=['GET'])
def list_blocks():
    # Obtener todas las carpetas en el directorio actual
    block_folders = [d for d in os.listdir('.') if os.path.isdir(d) and not d.startswith('.') and not d.startswith('__') and d not in ['venv', 'node_modules', 'build', 'public', 'src', 'uploads']]
    return jsonify({
        'blocks': block_folders,
        'total_blocks': len(block_folders)
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No se encontró ninguna imagen'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ninguna imagen'}), 400
    
    try:
        # Guardar la imagen temporalmente
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Usar el modelo PyTorch
        predicted_block, confidence, top_3_predictions = predict_block_pytorch(
            pytorch_model, class_names, filepath
        )
        
        # Eliminar la imagen temporal
        os.remove(filepath)
        
        return jsonify({
            'predicted_block': predicted_block,
            'confidence': float(confidence),
            'top_3_predictions': top_3_predictions
        })
        
    except Exception as e:
        print(f"Error durante la predicción: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/train')
def train():
    def event_stream():
        q = queue.Queue()
        def progress_callback(msg):
            q.put(msg)
        def run_training():
            # Implementa la lógica para entrenar el modelo PyTorch aquí
            q.put('DONE')
        threading.Thread(target=run_training).start()
        while True:
            msg = q.get()
            if msg == 'DONE':
                yield f'data: Entrenamiento finalizado\n\n'
                break
            yield f'data: {msg}\n\n'
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/training-plot')
def training_plot():
    return send_from_directory('static', 'training_history_pytorch.png')

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Ruta no encontrada'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Error interno del servidor'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 