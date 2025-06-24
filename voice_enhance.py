import gradio as gr
import numpy as np
import librosa
import joblib
import soundfile as sf
import os 
from feature_Extraction import extract_features

# Loading Model Components
svm_model = joblib.load('best_svm_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# Creating Output Directory
os.makedirs('outputs', exist_ok=True)

def apply_tremolo(y, sr, rate=8):
    t = np.linspace(0, len(y)/sr, len(y))
    mod = 0.5 * (1 + np.sin(2*np.pi * rate * t))
    return y * mod

def enhance_voice(y, sr, emotion, mode='emotion', voice_type=None):
    y_mod = y

    if mode == 'emotion':
        if emotion == 'happy':
            y_mod = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2)
            y_mod = librosa.effects.time_stretch(y = y_mod, rate=1.1)
        elif emotion == 'sad':
            y_mod = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-2)
            y_mod = librosa.effects.time_stretch(y = y_mod, rate=0.9)
        elif emotion == 'angry':
            y_mod = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=4)
            y_mod = librosa.effects.time_stretch(y = y_mod, rate=1.2)
        elif emotion == 'fearful':
            y_mod = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-3)
            y_mod = librosa.effects.time_stretch(y = y_mod, rate=0.8)
        elif emotion == 'disgust':
            y_mod = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-4)
            y_mod = librosa.effects.time_stretch(y = y_mod, rate=0.85)
        elif emotion == 'surprise':
            y_mod = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=5)
            y_mod = librosa.effects.time_stretch(y = y_mod, rate=1.3)
        elif emotion == 'neutral':
            y_mod = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=5)
            y_mod = librosa.effects.time_stretch(y = y_mod, rate=1.3)
    elif mode == "voice":
        if voice_type == 'robot':
            y_mod = apply_tremolo(y, sr, rate=15)
        elif voice_type == 'chipmunk':
            y_mod = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=10)
            y_mod = librosa.effects.time_stretch(y = y_mod,rate= 1.4)
        elif voice_type == 'deep':
            y_mod = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-6)
        elif voice_type == 'girl':
            y_mod = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=5)
        elif voice_type == 'boy':
            y_mod = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-5)
        else:
            print('Unknown voice type, returning original audio')

    return y_mod

def process_audio_gr(audio, mode, selected_emotion, voice_type):
    y, sr = librosa.load(audio, sr=None)

    # If emotion mode: detect from model
    if mode == 'emotion':
        features = extract_features(y, sr)
        features_scaled = scaler.transform([features])
        preds = svm_model.predict(features_scaled)
        detected_emotion = le.inverse_transform(preds)[0]
    else:
        detected_emotion = selected_emotion  # Use selected for display

    # Apply enhancement
    y_mod = enhance_voice(y, sr, emotion=selected_emotion, mode=mode, voice_type=voice_type)

    output_path = 'outputs/enhanced_output.wav'
    sf.write(output_path, y_mod, sr)

    return output_path, f"🎯 Detected Emotion: {detected_emotion}"

# Gradio UI
interface = gr.Interface(
    fn=process_audio_gr,
    inputs=[
        gr.Audio(sources='microphone', type='filepath', label='🎙️ Record Your Voice'),
        gr.Radio(['emotion', "voice"], label="Mode"),
        gr.Dropdown(['happy', 'sad', 'fearful', 'angry', 'neutral', 'disgust', 'surprise'], label="Select Emotion"),
        gr.Dropdown(['robot', 'girl', 'boy', 'chipmunk', 'deep'], label="Voice Type")
    ],
    outputs=[
        gr.Audio(type='filepath', label='🔊 Enhanced Audio'),
        gr.Text(label="🧠 Detected Emotion")
    ],
    title='🎛️ Emotion-Aware Voice Enhancer',
    description='Record your voice and get an enhanced version based on detected emotion or custom voice transformation.'
)

interface.launch()
