
import os#操作系统相关的操作
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from utils.audio_ser import SpeechEmotionRecognizer, EMOTION_ORDER as AUD_EMOS#自定义模块
from utils.face_recog import VideoFaceEmotion, EMOTION_ORDER as VID_EMOS
from utils.text_ser import TextSentimentRecognizer
from utils.multimodal_ser import MultimodalSentimentRecognizer
import subprocess#运行外部命令
import shutil#高级文件操作
import uuid#生成唯一的文件名
import librosa
import soundfile as sf

def convert_audio_to_wav(src_path, target_sr=16000):
    """把任意音频转成 WAV(16k/mono/PCM_s16le)，返回转换后的新路径。
    优先使用 ffmpeg；若不可用，则使用 librosa+soundfile 作为兜底。
    """
    base, ext = os.path.splitext(src_path)
    if ext.lower() == ".wav":#文件扩展名是这个
        return src_path

    dst_path = f"{base}_{uuid.uuid4().hex}.wav"

    # 方案一：ffmpeg
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-ac", "1",
            "-ar", str(target_sr),
            "-c:a", "pcm_s16le",
            dst_path
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return dst_path
        except Exception:
            # 回落到 librosa
            pass

    # 方案二：librosa + soundfile 兜底
    try:
        y, sr = librosa.load(src_path, sr=target_sr, mono=True)
        #加载源音频文件，目标采样率，单声道
        sf.write(dst_path, y, target_sr, subtype="PCM_16")
        #使用librosa.load方法加载源音频文件src_path，指定目标采样率target_sr，
        #并且设置为单声道（mono=True）。加载后，音频数据赋值给y，
        # 实际采样率赋值给sr
        return dst_path
    except Exception as e:
        raise RuntimeError(f"音频转换失败（需要 ffmpeg 或 librosa/soundfile）: {e}")

UPLOAD_AUDIO_DIR = "uploads/audio"#加载到文件uploads处
UPLOAD_VIDEO_DIR = "uploads/video"
UPLOAD_TEXT_DIR  = "uploads/text"

app = Flask(__name__)
#创建一个flask实例
app.secret_key = "dev-secret"
#进行数据加密

# Init models (look in ./models by default)
ser = SpeechEmotionRecognizer(models_dir="models")
#表明这部分是初始化模型，默认从./models目录查找模型相关文件。
vfer = VideoFaceEmotion(models_dir="models")
text_ser = TextSentimentRecognizer(models_dir="models")
multimodal_ser = MultimodalSentimentRecognizer(models_dir="models")

def _to_percent_list(probs_ordered, labels_order):
    #probs_ordered有序的概率字典，键为标签，值为对应概率
    #labels_order前端期望的标签顺序列表
    """Return list of percentages aligned with the frontend's label order."""
    return [ round(float(probs_ordered.get(lbl,0.0))*100, 2) for lbl in labels_order ]
    #将其转换为浮点数后乘以100得到百分比，再保留两位小数，最终返回由这些百分比组成的列表。
    #这样处理后的数据更便于前端展示不同情感标签的概率占比。

@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html")
#定义了一个名为index的视图函数。当触发上述路由时，Flask会调用这个函数来处理请求并返回响应。

@app.route("/analyze", methods=["POST"])
#@app.route(rule="/analyze", methods=["POST"])：
# 路由装饰器，rule="/analyze" 表示该路由对应的URL路径是/analyze；
# methods=["POST"]指定该路由接受 POST
#请求方法，POST常用于向服务器提交数据，比如表单提交、文件上传等场景。
def analyze():
    #Save uploads
    #定义了analyze函数，用于处理向/analyze路径发送的POST请求
    audio_path = None
    video_path = None
    text_path  = None
    text_content = None
    wav_path = None

    # audio
    # 1) 先保存上传的音频
    if "audio" in request.files and request.files["audio"].filename:
        a = request.files["audio"]
        audio_fn = secure_filename(a.filename)
        os.makedirs(UPLOAD_AUDIO_DIR, exist_ok=True)
        audio_path = os.path.join(UPLOAD_AUDIO_DIR, audio_fn)
        a.save(audio_path)

    # text - handle both file upload and direct text input
    if "text" in request.files and request.files["text"].filename:
        t = request.files["text"]
        text_fn = secure_filename(t.filename)
        text_path = os.path.join(UPLOAD_TEXT_DIR, text_fn)
        os.makedirs(UPLOAD_TEXT_DIR, exist_ok=True)
        t.save(text_path)
        # Read text content from file
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
        except Exception as e:
            text_content = f"Error reading file: {str(e)}"
    elif "text_input" in request.form and request.form["text_input"].strip():
        # Direct text input from form
        text_content = request.form["text_input"].strip()

    # video
    if "video" in request.files and request.files["video"].filename:
        v = request.files["video"]
        video_fn = secure_filename(v.filename)
        video_path = os.path.join(UPLOAD_VIDEO_DIR, video_fn)
        os.makedirs(UPLOAD_VIDEO_DIR, exist_ok=True)
        v.save(video_path)

    # Run models
    # (1) Speech emotion (Audio)
    aud_emo = "N/A"
    aud_probs = {k:0.0 for k in AUD_EMOS}
    if audio_path:
        try:
            wav_path = convert_audio_to_wav(audio_path, target_sr=16000)
            aud_res = ser.predict(wav_path)
            aud_emo = (aud_res.get("top_emotion","Unknown"))
            aud_probs = aud_res.get("probs", aud_probs)
        except Exception as e:
            print(f"[ANALYZE] Audio analysis failed: {e}")
            wav_path = None

    # (2) Face emotion (Video)
    vid_emo = "N/A"
    vid_probs = {k:0.0 for k in VID_EMOS}
    if video_path:
        vid_res = vfer.analyze(video_path)
        vid_emo = vid_res.get("top_emotion","Unknown")
        vid_probs = vid_res.get("probs", vid_probs)

    # (3) Text sentiment/personality
    text_trait = "N/A"
    text_traits = [0, 0, 0, 0, 0]
    if text_content:
        text_res = text_ser.predict(text_content)
        if not text_res:
            print("null")
        text_trait = text_res.get("top_emotion", "Unknown")
        text_probs = text_res.get("probs", {})
        # 前端模板顺序：Extraversion, Neuroticism, Agreeableness, Conscientiousness, Openness
        display_order = [
            "Extraversion",
            "Neuroticism",
            "Agreeableness",
            "Conscientiousness",
            "Openness",
        ]
        text_traits = [round(float(text_probs.get(label, 0.0)) * 100, 2) for label in display_order]
        print(text_traits)

    # (4) Multimodal fusion
    multimodal_result = None
    if (wav_path is not None) or text_content:
        multimodal_result = multimodal_ser.analyze_multimodal(
            audio_path=wav_path,
            text=text_content
        )

    # Map to the Jinja variables your dashboard uses
    # Audio section expects: emo, prob (list for 7 emotions)
    prob_audio_list = _to_percent_list(aud_probs, ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"])

    # Video section expects: emo, prob (again 7 emotions, different order)
    prob_video_list = _to_percent_list(vid_probs, ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])

    # For "others" distributions, provide simple mirrors or empty
    prob_audio_other = [round(100.0/7,2)]*7
    prob_video_other = [round(100.0/7,2)]*7

    # Integrated section — enhanced with multimodal results
    if multimodal_result and not multimodal_result.get("error"):
        fused_emotion = multimodal_result["fused"]["fused_emotion"]
        confidence = multimodal_result["fused"]["confidence"]
        integrated_emotion = f"融合结果：{fused_emotion} (置信度: {confidence:.2f})"
        integrated_summary = "基于多模态融合分析，综合音频、文本和视频信息得出结果。"
    else:
        integrated_emotion = f"音频：{aud_emo}；视频：{vid_emo}；文本：{text_trait}"
        integrated_summary = "基于当前上传的音频、视频与文本，已给出情绪识别结果。"

    # Enhanced trait analysis
    if text_trait != "N/A":
        integrated_trait_score = f"主要特质：{text_trait}"
        depression_description = "基于文本分析的人格特质评估。"
    else:
        integrated_trait_score = "N/A"
        depression_description = "文本分析暂未启用或分析失败。"

    ctx = {
        # 音频区（Speech）
        "aud_emo": aud_emo,
        "aud_prob": prob_audio_list,
        "aud_emo_other": "—",
        "aud_prob_other": prob_audio_other,

        # 视频区（Video / Face）
        "vid_emo": vid_emo,
        "vid_prob": prob_video_list,
        "vid_emo_other": "—",
        "vid_prob_other": prob_video_other,

        # 文本区（Enhanced）
        "trait": text_trait,
        "traits": text_traits,
        "common_words": [],
        "probas_others": [round(100.0/5,2)]*5,  # 5 personality traits
        "trait_others": "—",
        "common_words_others": [],

        # 综合区
        "integrated_emotion": integrated_emotion,
        "integrated_trait_score": integrated_trait_score,
        "depression_level": "N/A",
        "depression_description": depression_description,
        "integrated_summary": integrated_summary
    }
    return render_template("integrated_dash.html", **ctx)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
