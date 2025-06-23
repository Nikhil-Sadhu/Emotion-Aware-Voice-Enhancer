# 🎙️ Emotion-Aware Voice Enhancer

An AI-powered web app that detects emotions in voice recordings and transforms the voice to amplify or modify those emotions. Built with Python, Machine Learning, and Gradio for real-time interaction.

---

## 🚀 Features

- 🎯 **Emotion Detection** from voice using a trained SVM classifier
- 🎛️ **Emotion-based Voice Enhancement** (e.g., make happy voices sound happier)
- 🤖 **Voice Changer Modes** (robot, chipmunk, boy/girl voice, etc.)
- 🎤 **Record from Microphone** in-browser and get instant results
- 🎧 **Play Enhanced Audio** and download it
- ⚙️ Powered by `librosa`, `scikit-learn`, `gradio`, `soundfile`, and `joblib`

---

## 🧠 How It Works

### 1. Feature Extraction
- Extract MFCCs, Chroma, Mel Spectrogram, ZCR, and RMSE from voice clips.

### 2. Emotion Classification
- Trained an SVM model on voice data with emotional labels.
- Saved model and scaler using `joblib`.

### 3. Voice Transformation
- Apply pitch shifting, time stretching, and effects like tremolo.
- Choose between:
  - `Emotion Mode`: enhances detected emotion (happy, sad, angry, etc.)
  - `Voice Mode`: transforms voice style (robot, girl, boy, chipmunk, etc.)

### 4. Web Interface
- Built with Gradio: record → enhance → play → download.

---

## 🌐 Live Demo

> Coming soon on [Hugging Face Spaces](https://huggingface.co/spaces/) (optional)

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/emotion-voice-enhancer.git
cd emotion-voice-enhancer
pip install -r requirements.txt
