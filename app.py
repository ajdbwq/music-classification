import os
import traceback
from flask import Flask, request, jsonify, render_template
from inference import MusicGenreClassifier


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化分类器
classifier = MusicGenreClassifier(
    model_path="model/best_model.pth",
    class_map_path="model/class_map.json",
    scaler_path="model/feature_scaler.joblib"
)

@app.route('/')
def index():
    return render_template('index.html')

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
        # 执行预测
        result = classifier.predict(filepath)
        os.remove(filepath)  # 清理临时文件
        
        # 格式化概率数据
        sorted_probs = sorted(
            result['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]  # 只显示Top3
        
        return jsonify({
            'genre': result['prediction'],
            'probabilities': {k: f"{v*100:.1f}%" for k, v in sorted_probs}
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'处理失败: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)