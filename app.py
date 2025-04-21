import os
import random
import logging
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from inference import MusicGenreClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 读取 metadata.csv 文件
metadata = pd.read_csv('database/metadata.csv')

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

        # 获取预测的流派
        predicted_genre = result['prediction']
        
        # 筛选出相同流派的歌曲
        same_genre_songs = metadata[metadata['genre'] == predicted_genre]
        
        # 随机选择8首歌曲
        recommended_songs = same_genre_songs.sample(min(8, len(same_genre_songs)))[['title', 'duration', 'artist']].to_dict(orient='records')

        return jsonify({
            'genre': result['prediction'],
            'probabilities': [
                {"genre": item["genre"], "probability": f"{item['probability']*100:.1f}%"} 
                for item in result['probabilities'][:3]  # 直接取前3项
            ],
            'recommended_songs': recommended_songs,
        })
    
    except Exception as e:
        logging.error(f"处理预测请求时出错: {e}")
        return jsonify({'error': f'处理失败: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)