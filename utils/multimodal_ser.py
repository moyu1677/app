#!/usr/bin/env python3
"""
Multimodal Sentiment Recognition System
"""

import os
import numpy as np
from .audio_ser import SpeechEmotionRecognizer
from .text_ser import TextSentimentRecognizer

class MultimodalSentimentRecognizer:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.audio_recognizer = SpeechEmotionRecognizer(models_dir)
        self.text_recognizer = TextSentimentRecognizer(models_dir)
        
        # Fusion weights
        self.audio_weight = 0.4
        self.text_weight = 0.6
        
        print(f"[MULTIMODAL] Initialized with audio weight: {self.audio_weight}, text weight: {self.text_weight}")
    
    def analyze_audio(self, audio_path):
        """Analyze audio file for emotion"""
        try:
            if not os.path.exists(audio_path):
                return {"error": f"Audio file not found: {audio_path}"}
            result = self.audio_recognizer.predict(audio_path)
            return result
        except Exception as e:
            return {"error": f"Audio analysis failed: {str(e)}"}
    
    def analyze_text(self, text):
        """Analyze text for sentiment/personality"""
        try:
            if not text or not text.strip():
                return {"error": "Empty or invalid text input"}
            result = self.text_recognizer.predict(text)
            return result
        except Exception as e:
            return {"error": f"Text analysis failed: {str(e)}"}
    
    def analyze_multimodal(self, audio_path=None, text=None):
        """Main method for multimodal analysis"""
        results = {}
        
        # Analyze audio if provided
        if audio_path:
            print(f"[MULTIMODAL] Analyzing audio: {audio_path}")
            audio_result = self.analyze_audio(audio_path)
            results["audio"] = audio_result
        else:
            audio_result = {"error": "No audio provided"}
            results["audio"] = audio_result
        
        # Analyze text if provided
        if text:
            print(f"[MULTIMODAL] Analyzing text: {text[:50]}...")
            text_result = self.analyze_text(text)
            results["text"] = text_result
        else:
            text_result = {"error": "No text provided"}
            results["text"] = text_result
        
        # Fuse results
        fused_result = self.fuse_results(audio_result, text_result)
        results["fused"] = fused_result
        
        return results
    
    def fuse_results(self, audio_result, text_result):
        """Fuse results from multiple modalities"""
        try:
            fused_result = {
                "modalities": {"audio": audio_result, "text": text_result},
                "fused_emotion": "Unknown",
                "confidence": 0.0
            }
            
            # Check for errors
            audio_error = audio_result.get("error")
            text_error = text_result.get("error")
            
            if audio_error and text_error:
                fused_result["error"] = "Both modalities failed"
                return fused_result
            
            # Determine fused emotion
            if not audio_error and not text_error:
                # Both available - use text as primary (higher weight)
                fused_result["fused_emotion"] = text_result.get("top_emotion", "Unknown")
                fused_result["confidence"] = 0.8
            elif not audio_error:
                fused_result["fused_emotion"] = audio_result.get("top_emotion", "Unknown")
                fused_result["confidence"] = 0.6
            elif not text_error:
                fused_result["fused_emotion"] = text_result.get("top_emotion", "Unknown")
                fused_result["confidence"] = 0.7
            
            return fused_result
            
        except Exception as e:
            return {"error": f"Fusion failed: {str(e)}"}
    
    def get_available_modalities(self):
        """Get information about available modalities"""
        return {
            "audio": {
                "available": self.audio_recognizer.clf is not None,
                "model_type": "SVM Classifier",
                "emotions": ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
            },
            "text": {
                "available": self.text_recognizer.model is not None,
                "model_type": "Keras CNN-LSTM",
                "traits": ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
            }
        }
