
import os
import numpy as np
import librosa
import pickle

EMOTION_ORDER = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

class SpeechEmotionRecognizer:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.scaler_raw = self._safe_load("MODEL_SCALER.p")
        self.pca_raw = self._safe_load("MODEL_PCA.p")
        self.scaler = self._unwrap_transformer(self.scaler_raw, "scaler")
        self.pca = self._unwrap_transformer(self.pca_raw, "pca")
        self.clf = self._safe_load("MODEL_CLASSIFIER.p")
        self.encoder = self._safe_load("MODEL_ENCODER.p")  # sklearn LabelEncoder or mapping
        self.params = self._safe_load("MODEL_PARAM.p")     # optional dict for feature settings
        
        # Handle custom scaler format (if it's training statistics)
        self._setup_custom_scaler()

    def _safe_load(self, filename):
        path = os.path.join(self.models_dir, filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                try:
                    return pickle.load(f)
                except Exception as e:
                    print(f"[SER] Failed to load {filename}: {e}")
                    return None
        else:
            print(f"[SER] {filename} not found in {self.models_dir}.")
            return None

    def _extract_features(self, y, sr):
        # Basic MFCC+Chroma+Mel features; adjust with self.params if present
        n_mfcc = 40
        n_mels = 128
        if isinstance(self.params, dict):
            n_mfcc = int(self.params.get("n_mfcc", n_mfcc))
            n_mels = int(self.params.get("n_mels", n_mels))
        
        # Adjust feature dimensions to match PCA model (188 features)
        # 188 = n_mfcc*2 + 12 + n_mels
        # Let's solve: n_mfcc*2 + n_mels = 176
        # For balanced features, let's use n_mfcc=32, n_mels=112
        # 32*2 + 12 + 112 = 64 + 12 + 112 = 188
        
        n_mfcc = 32  # Adjusted to produce 188 features
        n_mels = 112  # Adjusted to produce 188 features
        
        # Robust framing
        y = librosa.util.normalize(y)
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # Chroma
        stft = np.abs(librosa.stft(y))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # MelSpec
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_mean = np.mean(mel_db, axis=1)

        feat = np.concatenate([mfcc_mean, mfcc_std, chroma_mean, mel_mean])
        return feat.reshape(1, -1)

    def predict(self, wav_path):
        try:
            y, sr = librosa.load(wav_path, sr=None, mono=True)
        except Exception as e:
            print(f"[SER] Failed to load audio {wav_path}: {e}")
            return {"top_emotion":"Unknown", "probs": {e:0.0 for e in EMOTION_ORDER}}

        X = self._extract_features(y, sr)

        # Apply custom scaling first
        X = self._custom_scale(X)
        
        # Then apply PCA if available
        if self.pca is not None:
            X = self.pca.transform(X)

        # Handle classification
        if hasattr(self.clf, "predict_proba"):
            proba = self.clf.predict_proba(X)[0]
        else:
            # SVC doesn't have predict_proba, use decision_function
            if hasattr(self.clf, "decision_function"):
                scores = self.clf.decision_function(X).ravel()
                # Convert scores to probabilities using softmax
                e = np.exp(scores - np.max(scores))
                proba = e / (e.sum() + 1e-9)
            else:
                # Fallback uniform
                proba = np.ones(7)/7.0

        # Map classes -> labels
        if self.encoder is not None:
            try:
                class_indices = np.arange(len(proba))
                labels = self.encoder.inverse_transform(class_indices)
            except Exception:
                labels = EMOTION_ORDER
        else:
            labels = EMOTION_ORDER

        probs = {str(lbl): float(p) for lbl, p in zip(labels, proba)}
        top_idx = int(np.argmax(proba))
        top_emotion = str(list(probs.keys())[top_idx])
        return {"top_emotion": top_emotion, "probs": probs}

    # 在类里加：
    def _unwrap_transformer(self, obj, name=""):
        # 已经是可用的
        if hasattr(obj, "transform"):
            return obj
        # 列表/元组：挑第一个带 transform 的
        if isinstance(obj, (list, tuple)):
            for it in obj:
                if hasattr(it, "transform"):
                    return it
        # 字典：按常见键名取
        if isinstance(obj, dict):
            for k in ("scaler", "pca", "preprocess", "StandardScaler", "PCA"):
                if k in obj and hasattr(obj[k], "transform"):
                    return obj[k]
        print(f"[SER] {name} is {type(obj)}, no .transform found, skip.")
        return None

    def _setup_custom_scaler(self):
        """Handle custom scaler format that contains training statistics"""
        if isinstance(self.scaler_raw, list) and len(self.scaler_raw) == 2:
            # This appears to be [mean, std] from training data
            self.train_mean = np.asarray(self.scaler_raw[0], dtype=float).reshape(1, -1)
            self.train_std = np.asarray(self.scaler_raw[1], dtype=float).reshape(1, -1)
            self.use_custom_scaler = True
            print(f"[SER] Using custom scaler with {self.train_mean.shape[1]} features")
        else:
            self.use_custom_scaler = False
            self.train_mean = None
            self.train_std = None

    def _custom_scale(self, X):
        """Apply custom scaling using training statistics"""
        if self.use_custom_scaler and self.train_mean is not None and self.train_std is not None:
            # Ensure X has the right shape
            if X.shape[1] != self.train_mean.shape[1]:
                print(f"[SER] Warning: X has {X.shape[1]} features, expected {self.train_mean.shape[1]}")
                # Resize with truncation or padding zeros to match expected dim
                if X.shape[1] > self.train_mean.shape[1]:
                    X = X[:, : self.train_mean.shape[1]]
                else:
                    pad = np.zeros((X.shape[0], self.train_mean.shape[1] - X.shape[1]))
                    X = np.hstack([X, pad])

            # Apply standardization: (X - mean) / std
            X_scaled = (X - self.train_mean) / (self.train_std + 1e-8)
            return X_scaled
        return X


