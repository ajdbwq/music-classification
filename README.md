# 音乐流派分类系统

## 一、项目概述

本项目是一个基于 Python 和 Flask 框架构建的音乐流派分类系统。用户可以上传音频文件（支持 WAV、MP3、OGG 格式），系统会对音频进行特征提取和分析，预测其所属的音乐流派，并提供相似流派的歌曲推荐。

## 二、项目结构



```
music-classification/

├── app.py              # Flask应用主文件，处理HTTP请求和路由

├── inference.py        # 音频特征处理和模型推理模块

├── model/              # 存储训练好的模型文件和相关配置

│   ├── best\_model.pth  # 训练好的模型权重

│   ├── class\_map.json  # 类别映射文件，将模型输出的类别编号映射为具体的音乐流派名称

│   ├── feature\_scaler.joblib  # 特征标准化器，用于对提取的音频特征进行标准化处理

│   └── feature\_order.json  # 特征顺序文件，规定了特征提取的顺序

├── database/           # 音乐元数据文件

│   └── metadata.csv    # 包含音乐的标题、时长、艺术家和流派等信息，用于相似歌曲推荐

├── uploads/            # 上传音频文件的临时存储目录

├── static/             # 静态资源文件

│   ├── css/            # 样式表文件

│   │   └── style.css   # 定义网页的样式

│   └── js/             # JavaScript文件

│       ├── app.js      # 处理网页上的交互逻辑，如文件上传、进度条显示和结果展示

│       └── particles.js  # 实现粒子效果的脚本

└── templates/          # HTML模板文件

&#x20;   └── index.html      # 项目主页模板，提供用户交互界面
```

## 三、安装步骤

### 1. 克隆项目



```
git clone \[项目仓库地址]

cd music-classification
```

### 2. 创建并激活虚拟环境（可选但推荐）

#### Windows



```
python -m venv venv

.\venv\Scripts\activate
```

#### Linux/Mac



```
python -m venv venv

source venv/bin/activate
```

### 3. 安装依赖



```
pip install -r requirements.txt
```

`requirements.txt` 文件应包含以下依赖：



```
Flask==2.2.3

librosa==0.10.0.post2

numpy==1.24.3

torch==2.0.1

pandas==2.0.1

joblib==1.2.0
```

## 四、运行项目



```
python app.py
```

应用启动后，在浏览器中访问 `http://localhost:5000` 即可打开项目主页。

## 五、使用方法

打开项目主页，点击 “点击选择音乐文件（支持 WAV/MP3）” 或直接将音频文件拖放到指定区域。

选择要分类的音频文件后，点击 “开始识别” 按钮。

系统会自动上传音频文件，提取音频特征，使用预训练模型进行预测，并显示预测的音乐流派、概率分布以及相似歌曲推荐。