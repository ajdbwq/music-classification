import librosa
import numpy as np
import torch
import json
from torch import nn
from joblib import load


# 配置参数
SAMPLE_RATE = 22050
DURATION = 30
N_MFCC = 40
HOP_LENGTH = 512
N_FFT = 2048
DROPOUT_RATE = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CSV特征分类器
class CSVFeatureClassifier(nn.Module):
    """CSV特征分类器：处理表格特征"""
    def __init__(self, input_size, num_classes):
        super().__init__()
        # 添加批归一化和更多的丢弃层
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE * 0.8),  # 逐渐减少丢弃率
            
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

class FeatureProcessor:
    """音频特征处理器（集成特征提取与标准化）"""
    def __init__(self, scaler_path):
        # 加载训练时的标准化参数
        self.expected_features = 58  # 根据实际CSV特征列数调整
        self.scaler = self._load_scaler(scaler_path)

    def _load_scaler(self, scaler_path):
        """加载训练时的标准化器"""
        try:
            # 加载保存的scaler对象
            scaler = load(scaler_path)
            
            # 验证特征维度
            if scaler.mean_.shape[0] != self.expected_features:
                raise ValueError(f"标准化器维度不匹配，预期{self.expected_features}，实际{scaler.mean_.shape[0]}")
                
            return scaler
            
        except FileNotFoundError:
            raise FileNotFoundError(f"未找到标准化器文件: {scaler_path}")
        except Exception as e:
            raise RuntimeError(f"加载标准化器失败: {str(e)}")

    def process(self, audio_path):
        """完整特征处理流程"""
        # 加载音频
        audio = self._load_audio(audio_path)
        
        # 提取特征统计量
        features = self._extract_features(audio)
        
        # 标准化处理
        return self._normalize_features(features)

    def _load_audio(self, path):
        """标准化音频加载"""
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
        if len(audio) < SAMPLE_RATE * DURATION:
            audio = np.pad(audio, (0, SAMPLE_RATE * DURATION - len(audio)))
        else:
            audio = audio[:SAMPLE_RATE * DURATION]
        return audio

    def _extract_features(self, audio):
        """提取58个音频特征"""
        features = {}
        
        # 1. length (样本数)
        features['length'] = len(audio)
        
        # 2-3. chroma_stft
        chroma = librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
        features['chroma_stft_mean'] = np.mean(chroma)
        features['chroma_stft_var'] = np.var(chroma)
        
        # 4-5. RMS能量
        rms = librosa.feature.rms(y=audio, frame_length=N_FFT, hop_length=HOP_LENGTH)
        features['rms_mean'] = np.mean(rms)
        features['rms_var'] = np.var(rms)
        
        # 6-11. 频谱特征
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
        
        features.update({
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_centroid_var': np.var(spectral_centroid),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_bandwidth_var': np.var(spectral_bandwidth),
            'rolloff_mean': np.mean(spectral_rolloff),
            'rolloff_var': np.var(spectral_rolloff)
        })
        
        # 12-13. 过零率
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=N_FFT, hop_length=HOP_LENGTH)
        features['zero_crossing_rate_mean'] = np.mean(zcr)
        features['zero_crossing_rate_var'] = np.var(zcr)
        
        # 14-17. 和声与感知特征
        y_harm = librosa.effects.harmonic(audio)  # 谐波部分
        y_perc = librosa.effects.percussive(audio)  # 冲击部分
        
        features.update({
            'harmony_mean': np.mean(y_harm),
            'harmony_var': np.var(y_harm),
            'perceptr_mean': np.mean(y_perc),
            'perceptr_var': np.var(y_perc)
        })
        
        # 18. 节奏
        tempo = librosa.beat.tempo(y=audio, sr=SAMPLE_RATE)[0]
        features['tempo'] = tempo
        
        # 19-58. MFCC特征（20个系数，每个均值和方差）
        mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, 
                                    n_mfcc=20, n_fft=N_FFT, hop_length=HOP_LENGTH)
        for i in range(20):
            features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc{i+1}_var'] = np.var(mfccs[i])
        
        # 验证特征数量
        assert len(features) == 58, f"特征数量错误: {len(features)}"
        
        return features

    def _normalize_features(self, features):
        """特征标准化"""
        # 按训练时的特征顺序构建数组
        feature_array = np.array([features[k] for k in sorted(features.keys())])
        return self.scaler.transform(feature_array.reshape(1, -1))

class MusicGenreClassifier:
    def __init__(self, model_path, class_map_path, scaler_path):
        # 初始化特征处理器
        self.feature_processor = FeatureProcessor(scaler_path)
        
        # 加载类别映射
        with open(class_map_path) as f:
            self.classes = json.load(f)
        
        # 初始化模型
        self.model = CSVFeatureClassifier(58, num_classes=len(self.classes))
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.to(DEVICE)
        self.model.eval()

    def predict(self, audio_path):
        """端到端预测流程"""
        # 特征处理
        processed_features = self.feature_processor.process(audio_path)
        
        # 转换为张量
        input_tensor = torch.tensor(processed_features, dtype=torch.float32).to(DEVICE)
        
        # 推理预测
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # 生成预测结果
        return {
            "prediction": self.classes[np.argmax(probs)],
            "probabilities": {cls: float(prob) for cls, prob in zip(self.classes, probs)}
        }

if __name__ == "__main__":
    classifier = MusicGenreClassifier(
        model_path="model/best_model.pth",
        class_map_path="model/class_map.json",
        scaler_path="model/feature_scaler.joblib"
    )
    
    result = classifier.predict("blues.00001.wav")
    print(f"预测结果: {result['prediction']}")
    print("概率分布:")
    for genre, prob in result['probabilities'].items():
        print(f"{genre}: {prob:.2%}")