# This model is designed to test the basic functionality of emotion detection.  
# After verifying its performance, the next step will be to integrate a text-to-speech (TTS) module  
# to provide real-time verbal feedback based on detected emotions. 

import cv2
import torch
from fer import FER
from transformers import (
    Wav2Vec2ForSequenceClassification, 
    Wav2Vec2Processor, 
    pipeline, 
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig
)
import sounddevice as sd
import numpy as np

#models
emotion_detector = FER()
model_name = "facebook/wav2vec2-base-960h"
wav2vec_processor = Wav2Vec2Processor.from_pretrained(model_name)
wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# Text emotion recognition model
text_model_name = "SamLowe/roberta-base-go_emotions"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModelForSequenceClassification.from_pretrained(text_model_name)
classifier = pipeline("sentiment-analysis", model=text_model, tokenizer=text_tokenizer)

# DeepSeek model for complex emotions
llm_model_name = "deepseek-ai/DeepSeek-R1-Zero"
llm_config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True)
if hasattr(llm_config, "quantization_config"):
    llm_config.quantization_config = None

llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, config=llm_config, trust_remote_code=True)

def analyze_complex_emotion(face_emotion, speech_emotion, text_emotion):
    """
    Combine the detected emotions and use the language model to infer complex emotions.
    """
    input_text = (f"Facial Emotion: {face_emotion}, Speech Emotion: {speech_emotion}, "
                  f"Text Emotion: {text_emotion}. What is the overall complex emotion?")
    inputs = llm_tokenizer(input_text, return_tensors="pt")
    outputs = llm_model.generate(**inputs, max_length=50)
    complex_emotion = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return complex_emotion

# Function to record audio
def record_audio(duration=3, samplerate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.float32)
    sd.wait()
    return np.squeeze(audio)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect facial emotions
    emotions = emotion_detector.detect_emotions(frame)
    face_emotion = "Neutral"
    if emotions:
        (x, y, w, h) = emotions[0]['box']
        face_emotion, score = max(emotions[0]['emotions'].items(), key=lambda item: item[1])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{face_emotion}: {score:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Process audio for speech emotion recognition
    audio_data = record_audio()
    inputs_audio = wav2vec_processor(audio_data, return_tensors="pt", sampling_rate=16000, padding=True)
    with torch.no_grad():
        logits = wav2vec_model(**inputs_audio).logits
    speech_emotion = torch.argmax(logits, dim=-1).item()
    
    # Analyze text emotion (replace with real-time text input if desired)
    text_emotions = classifier("Example text input")
    text_emotion = text_emotions[0]['label']
    
    # Compute overall complex emotion
    complex_emotion = analyze_complex_emotion(face_emotion, speech_emotion, text_emotion)
    
    # Display results
    cv2.putText(frame, f"Speech Emotion: {speech_emotion}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Text Emotion: {text_emotion}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Complex Emotion: {complex_emotion}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('Multimodal Emotion Recognition', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

