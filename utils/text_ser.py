#!/usr/bin/env python3
"""
Text Sentiment Recognition using Keras Neural Network
"""

import os
import json
import numpy as np
import pickle

# Fix imports for newer TensorFlow versions
try:
    from keras.models import model_from_json
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
except ImportError:
    try:
        from tensorflow import keras
        from keras.models import model_from_json
        from keras.preprocessing.text import Tokenizer
        from keras.preprocessing.sequence import pad_sequences
    except ImportError:
        from tensorflow.keras.models import model_from_json
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences

class TextSentimentRecognizer:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.model = None
        self.tokenizer = None
        self.max_sequence_length = 300
        self.vocab_size = 27677
        
        # Load the Keras model
        self._load_model()
        
        # Initialize tokenizer
        self._setup_tokenizer()
        
        # Define emotion labels (based on the model output)
        self.emotion_labels = [
            "Openness",      # 开放性
            "Conscientiousness",  # 尽责性
            "Extraversion",  # 外向性
            "Agreeableness", # 宜人性
            "Neuroticism"    # 神经质
        ]
    
    def _load_model(self):
        """Load the Keras model from JSON and weights files"""
        try:
            # Load model architecture
            model_json_path = os.path.join(self.models_dir, "Personality_traits_NN.json")
            weights_path = os.path.join(self.models_dir, "Personality_traits_NN.weights.h5")
            
            if not os.path.exists(model_json_path):
                print(f"[TEXT_SER] Model JSON not found: {model_json_path}")
                return
                
            if not os.path.exists(weights_path):
                print(f"[TEXT_SER] Model weights not found: {weights_path}")
                return
            
            with open(model_json_path, 'r') as json_file:
                model_json = json_file.read()
            
            self.model = model_from_json(model_json)
            self.model.load_weights(weights_path)
            
            # Compile the model
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"[TEXT_SER] Model loaded successfully")
            
        except Exception as e:
            print(f"[TEXT_SER] Failed to load model: {e}")
            self.model = None
    
    def _setup_tokenizer(self):
        """Setup text tokenizer for preprocessing"""
        try:
            self.tokenizer = Tokenizer(num_words=self.vocab_size)
            # We'll need to fit the tokenizer on some text data
            # For now, we'll create a basic one
            print(f"[TEXT_SER] Tokenizer initialized")
        except Exception as e:
            print(f"[TEXT_SER] Failed to setup tokenizer: {e}")
            self.tokenizer = None
    
    def _preprocess_text(self, text):
        """Preprocess text for model input"""
        if self.tokenizer is None:
            print("[TEXT_SER] Tokenizer not available")
            return None
        
        try:
            # Clean and tokenize text
            text = str(text).lower().strip()
            
            # Create a simple tokenization approach for now
            # Split text into words and create a simple vocabulary
            words = text.split()
            
            # Create a simple word-to-index mapping
            word_to_index = {}
            for i, word in enumerate(words, 1):
                if word not in word_to_index:
                    word_to_index[word] = i
            
            # Convert text to sequence using simple mapping
            sequence = []
            for word in words:
                if word in word_to_index:
                    sequence.append(word_to_index[word])
                else:
                    sequence.append(0)  # Unknown word
            
            # Pad or truncate to max length
            if len(sequence) < self.max_sequence_length:
                sequence.extend([0] * (self.max_sequence_length - len(sequence)))
            else:
                sequence = sequence[:self.max_sequence_length]
            
            # Convert to numpy array and reshape
            sequence = np.array(sequence, dtype=np.float32)
            return sequence.reshape(1, -1)
            
        except Exception as e:
            print(f"[TEXT_SER] Text preprocessing failed: {e}")
            return None
    
    def predict(self, text):
        """Predict sentiment from text"""
        if self.model is None:
            return {
                "top_emotion": "Unknown",
                "probs": {label: 0.0 for label in self.emotion_labels},
                "error": "Model not loaded"
            }
        
        try:
            # Preprocess text
            X = self._preprocess_text(text)
            if X is None:
                return {
                    "top_emotion": "Unknown", 
                    "probs": {label: 0.0 for label in self.emotion_labels},
                    "error": "Text preprocessing failed"
                }
            
            # Make prediction
            predictions = self.model.predict(X, verbose=0)
            
            # Convert predictions to probabilities
            probs = {}
            for i, label in enumerate(self.emotion_labels):
                probs[label] = float(predictions[0][i])
            
            # Find top emotion
            top_idx = np.argmax(predictions[0])
            top_emotion = self.emotion_labels[top_idx]
            
            return {
                "top_emotion": top_emotion,
                "probs": probs
            }
            
        except Exception as e:
            print(f"[TEXT_SER] Prediction failed: {e}")
            return {
                "top_emotion": "Unknown",
                "probs": {label: 0.0 for label in self.emotion_labels},
                "error": str(e)
            }
    
    def predict_batch(self, texts):
        """Predict sentiment for multiple texts"""
        if self.model is None:
            return [{"top_emotion": "Unknown", "error": "Model not loaded"} for _ in texts]
        
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        
        return results
