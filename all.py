import torch
import sounddevice as sd  
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC  
from transformers import BertTokenizer, BertForSequenceClassification  
from transformers import pipeline 
import pyttsx3 
from fer import FER  
import cv2 

# Initialize Wav2Vec2 for audio-to-text transcription
wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
wav2vec2_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Initialize BERT for text-based emotion analysis
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=7)

# Load fine-tuned BERT for complex emotions
complex_emotion_analyzer = pipeline("text-classification", model="finiteautomata/bertweet-base-sentiment-analysis")

tts_engine = pyttsx3.init()
emotion_detector = FER()
emotion_labels = {0: "anger", 1: "disgust", 2: "fear", 3: "joy", 4: "neutral", 5: "sadness", 6: "surprise"}

# Function to record audio
def record_audio(duration=10, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording finished.")
    return audio.flatten()

# Function to convert audio to text using Wav2Vec2
def audio_to_text(audio, sample_rate=16000):
    inputs = wav2vec2_processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        logits = wav2vec2_model(**inputs).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = wav2vec2_processor.batch_decode(predicted_ids)[0]
    return transcription

# Function to analyze text for basic emotions using BERT
def analyze_text_for_emotions(text):
    # Tokenize text
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Perform inference
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    # Get predicted emotion
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    predicted_emotion = emotion_labels[predicted_class]
    
    return predicted_emotion

# Function to analyze text for complex emotions using fine-tuned BERT
def analyze_complex_emotions(text):
    result = complex_emotion_analyzer(text)
    return result[0]['label']

# Function to analyze facial expressions
def analyze_facial_emotions(frame):
    emotions = emotion_detector.detect_emotions(frame)
    if emotions:
        dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
        return dominant_emotion
    return None

# Function to provide feedback using text-to-speech
def provide_feedback(feedback):
    print(feedback)
    tts_engine.say(feedback)
    tts_engine.runAndWait()

# Main loop for continuous audio and video analysis
def real_time_emotion_analysis():
    try:
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        while True:
            audio = record_audio(duration=10)
            
            # Convert audio to text
            transcription = audio_to_text(audio)
            print(f"Transcription: {transcription}")
            
            # Analyze text for basic emotions
            predicted_emotion = analyze_text_for_emotions(transcription)
            print(f"Predicted Basic Emotion: {predicted_emotion}")
            
            # Analyze text for complex emotions
            complex_emotion = analyze_complex_emotions(transcription)
            print(f"Predicted Complex Emotion: {complex_emotion}")
            
            # Capture video frame for facial emotion analysis
            ret, frame = cap.read()
            if ret:
                # Analyze facial emotions
                facial_emotion = analyze_facial_emotions(frame)
                print(f"Predicted Facial Emotion: {facial_emotion}")
            
            # Provide feedback based on detected emotions
            feedback = []
            feedback.append(f"Basic Emotion: {predicted_emotion}")
            feedback.append(f"Complex Emotion: {complex_emotion}")
            if facial_emotion:
                feedback.append(f"Facial Emotion: {facial_emotion}")
            
            # Combine feedback and provide it via text-to-speech
            combined_feedback = " ".join(feedback)
            provide_feedback(combined_feedback)
            
            # Wait for a short time before the next recording
            print("Waiting for the next recording...")
            sd.sleep(1000)
    
    except KeyboardInterrupt:
        print("Stopping real-time emotion analysis.")
    finally:
        cap.release()
        
real_time_emotion_analysis()