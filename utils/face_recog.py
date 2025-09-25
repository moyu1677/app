
import os
import cv2
import pickle
import numpy as np
try:
    import face_recognition
    FR_OK = True
except Exception as e:
    print("[Face] face_recognition import failed:", e)
    FR_OK = False

EMOTION_ORDER = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

class VideoFaceEmotion:
    """
    Face-based emotion recognition on video frames.
    Two modes:
      1) If Keras model `xception_fer2013.h5` is present in models_dir, use it on cropped faces.
      2) Else, return a safe placeholder distribution.
    """
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.model = None
        self.input_size = (48,48)
        self._load_keras_model()

    def _load_keras_model(self):
        path = os.path.join(self.models_dir, "facial recognition.h5")
        if os.path.exists(path):
            try:
                from tensorflow.keras.models import load_model
                self.model = load_model(path, compile=False)
                print("[Face] Loaded Keras FER model.")
            except Exception as e:
                print("[Face] Failed to load Keras model:", e)
                self.model = None
        else:
            print("[Face] Keras FER model file not found; using placeholder.")

    def _predict_face_emotion(self, face_img_gray):
        if self.model is None:
            # Placeholder: uniform distribution
            probs = np.ones(len(EMOTION_ORDER))/len(EMOTION_ORDER)
            return probs
        # Resize to model input
        img = cv2.resize(face_img_gray, self.input_size, interpolation=cv2.INTER_AREA)
        img = img.astype("float32")/255.0
        img = np.expand_dims(img, axis=(0,-1))  # (1,48,48,1)
        y = self.model.predict(img, verbose=0)[0]
        # Ensure shape correctness
        if y.shape[0] != len(EMOTION_ORDER):
            # pad/trim to 7
            y = cv2.resize(y.reshape(1,-1), (len(EMOTION_ORDER),1), interpolation=cv2.INTER_AREA).ravel()
        # softmax normalize
        y = np.maximum(y, 1e-9)
        y = y/np.sum(y)
        return y

    def analyze(self, video_path, sample_every=10):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Face] Cannot open video {video_path}")
            return {"top_emotion":"Unknown", "probs": {e:0.0 for e in EMOTION_ORDER}}

        frame_idx = 0
        agg = np.zeros(len(EMOTION_ORDER), dtype=float)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every != 0:
                frame_idx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces
            faces = []
            if FR_OK:
                # Use HOG-based detector via face_recognition (simple & light)
                locations = face_recognition.face_locations(frame, model="hog")
                for (top, right, bottom, left) in locations:
                    face = gray[top:bottom, left:right]
                    if face.size > 0:
                        faces.append(face)
            else:
                # Fallback: whole frame as a "face"
                faces.append(gray)

            for face in faces:
                probs = self._predict_face_emotion(face)
                agg += probs

            frame_idx += 1

        cap.release()
        if np.sum(agg) <= 0:
            agg = np.ones(len(EMOTION_ORDER))
        agg = agg/np.sum(agg)

        probs = {label: float(p) for label, p in zip(EMOTION_ORDER, agg)}
        top_emotion = EMOTION_ORDER[int(np.argmax(agg))]
        return {"top_emotion": top_emotion, "probs": probs}
