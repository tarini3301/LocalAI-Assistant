from faster_whisper import WhisperModel
from gtts import gTTS
import os
import pyttsx3

# Initialize model (tiny for fast, or medium/large for better accuracy)
model = WhisperModel("tiny", compute_type="int8")  # Options: "int8", "float16", "float32"

def transcribe_audio(file):
    segments, _ = model.transcribe(file)
    text = ""
    for segment in segments:
        text += segment.text + " "
    return text.strip()

def speak(text, method="offline"):
    if method == "offline":
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    else:
        tts = gTTS(text)
        tts.save("output.mp3")
        os.system("start output.mp3")  # Windows
