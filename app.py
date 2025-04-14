import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from inference import MusicGenreClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化分类器
classifier = MusicGenreClassifier(
    model_path="model/best_model.pth",
    class_map_path="model/class_map.json",
    scaler_path="model/feature_scaler.joblib",
    feature_order_path="model/feature_order.json"
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('.', path)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': '未选择文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '空文件名'}), 400
    
    # 保存临时文件
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    try:
        result = classifier.predict(filepath)
        os.remove(filepath)

        return jsonify({
            'genre': result['prediction'],
            'probabilities': [
                {"genre": item["genre"], "probability": f"{item['probability']*100:.1f}%"} 
                for item in result['probabilities'][:3]  # 直接取前3项
            ]
        })
    
    except Exception as e:
        return jsonify({'error': f'处理失败: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
