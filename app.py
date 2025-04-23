import os
import random
import logging
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from inference import MusicGenreClassifier
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 读取 music_features.csv 文件
music_features = pd.read_csv('database/music_features.csv')

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

        # 提取用户上传音频的特征
        user_features = result['features']
        
        # 筛选出所有音乐的特征
        all_features = music_features[["length", "chroma_stft_mean", "chroma_stft_var", "rms_mean", "rms_var", "spectral_centroid_mean", "spectral_centroid_var", "spectral_bandwidth_mean", "spectral_bandwidth_var", "rolloff_mean", "rolloff_var", "zero_crossing_rate_mean", "zero_crossing_rate_var", "harmony_mean", "harmony_var", "perceptr_mean", "perceptr_var", "tempo", "mfcc1_mean", "mfcc1_var", "mfcc2_mean", "mfcc2_var", "mfcc3_mean", "mfcc3_var", "mfcc4_mean", "mfcc4_var", "mfcc5_mean", "mfcc5_var", "mfcc6_mean", "mfcc6_var", "mfcc7_mean", "mfcc7_var", "mfcc8_mean", "mfcc8_var", "mfcc9_mean", "mfcc9_var", "mfcc10_mean", "mfcc10_var", "mfcc11_mean", "mfcc11_var", "mfcc12_mean", "mfcc12_var", "mfcc13_mean", "mfcc13_var", "mfcc14_mean", "mfcc14_var", "mfcc15_mean", "mfcc15_var", "mfcc16_mean", "mfcc16_var", "mfcc17_mean", "mfcc17_var", "mfcc18_mean", "mfcc18_var", "mfcc19_mean", "mfcc19_var", "mfcc20_mean", "mfcc20_var"]].values
        
        # 计算余弦相似度
        similarities = cosine_similarity([user_features], all_features)[0]
        
        # 获取相似度最高的8首音乐的索引
        top_indices = similarities.argsort()[-9:][::-1][1:]  # 排除自身
        
        # 获取推荐的歌曲信息
        recommended_songs = music_features.iloc[top_indices][['title', 'duration', 'artist']].to_dict(orient='records')

        # 为推荐的歌曲添加音频路径
        for song in recommended_songs:
            song_id = music_features[music_features['title'] == song['title']]['id'].values[0]
            song_folder = str(song_id).zfill(6)[:3]
            song['audio_path'] = f'database/fma_small/{song_folder}/{str(song_id).zfill(6)}.mp3'

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