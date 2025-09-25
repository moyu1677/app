import tensorflow as tf

# 假设要加载 facial_recognition.h5 文件
model_path = "models/face_recognition.h5"
model = tf.keras.models.load_model(model_path)
# 打印模型结构
model.summary()
