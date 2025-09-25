
# 多模态情感识别系统 (Multimodal Emotion Recognition System)

这是一个基于Flask的多模态情感识别系统，集成了音频、文本和视频分析功能。

## 功能特性

### 🎵 音频情感识别 (Audio Emotion Recognition)
- 支持多种音频格式：WAV, MP3, M4A, FLAC
- 基于SVM分类器的7种情感识别
- 情感类别：愤怒、厌恶、恐惧、快乐、中性、悲伤、惊讶

### 📝 文本情感/人格分析 (Text Sentiment/Personality Analysis)
- 基于Keras CNN-LSTM深度学习模型
- 5大人格特质分析：开放性、尽责性、外向性、宜人性、神经质
- 支持直接文本输入和文件上传

### 🎥 视频人脸情感识别 (Video Face Emotion Recognition)
- 基于OpenCV的人脸检测
- 实时情感分析
- 支持多种视频格式

### 🔄 多模态融合 (Multimodal Fusion)
- 智能融合多种模态的分析结果
- 加权融合算法
- 置信度评估

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   音频输入      │    │   文本输入      │    │   视频输入      │
│   (Audio)       │    │   (Text)        │    │   (Video)       │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ 音频情感识别    │    │ 文本人格分析    │    │ 人脸情感识别    │
│ (SVM + PCA)     │    │ (Keras CNN-LSTM)│    │ (OpenCV + ML)   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    多模态融合引擎       │
                    │  (Multimodal Fusion)   │
                    └─────────────┬───────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │     综合情感结果        │
                    │   (Integrated Result)   │
                    └─────────────────────────┘
```

## 安装和运行

### 环境要求
- Python 3.8+
- TensorFlow 2.15+
- Flask 3.0+
- 其他依赖见 `requirements.txt`

### 安装步骤
1. 克隆项目
```bash
git clone <repository-url>
cd multimodal_app
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 运行应用
```bash
python app.py
```

4. 访问系统
```
http://localhost:5000
```

## 模型文件

系统需要以下预训练模型文件（放在 `models/` 目录下）：

### 音频模型
- `MODEL_CLASSIFIER.p` - SVM分类器
- `MODEL_PCA.p` - PCA降维模型
- `MODEL_SCALER.p` - 特征标准化参数
- `MODEL_ENCODER.p` - 标签编码器
- `MODEL_PARAM.p` - 模型参数

### 文本模型
- `Personality_traits_NN.json` - Keras模型架构
- `Personality_traits_NN.weights.h5` - 模型权重

### 人脸识别模型
- `facial recognition.h5` - 人脸识别模型

## API接口

### 主要路由
- `GET /` : 上传页面
- `POST /analyze` : 多模态分析接口

### 分析接口参数
- `audio`: 音频文件
- `text`: 文本文件
- `text_input`: 直接文本输入
- `video`: 视频文件

## 使用说明

1. **音频分析**: 上传音频文件或使用在线录音功能
2. **文本分析**: 直接输入文本或上传文本文件
3. **视频分析**: 上传视频文件或使用在线录制功能
4. **多模态分析**: 同时使用多种模态获得更准确的结果

## 技术特点

- **模块化设计**: 各模态独立，易于扩展
- **智能融合**: 基于置信度的多模态结果融合
- **实时处理**: 支持在线录制和分析
- **跨平台**: 支持Windows、Linux、macOS
- **响应式UI**: 现代化的Web界面

## 开发说明

### 项目结构
```
multimodal_app/
├── app.py                 # 主应用文件
├── utils/                 # 工具模块
│   ├── audio_ser.py      # 音频情感识别
│   ├── text_ser.py       # 文本情感识别
│   ├── face_recog.py     # 人脸情感识别
│   └── multimodal_ser.py # 多模态融合
├── models/                # 模型文件
├── templates/             # HTML模板
├── static/                # 静态资源
└── uploads/               # 上传文件目录
```

### 扩展新模态
1. 在 `utils/` 目录下创建新的识别器类
2. 实现 `predict()` 方法
3. 在 `MultimodalSentimentRecognizer` 中集成
4. 更新前端界面

## 许可证

本项目采用MIT许可证。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！
