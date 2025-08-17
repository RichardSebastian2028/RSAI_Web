from flask import Flask, render_template, request, send_file, jsonify
import io, os, tempfile, wave, base64
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, UnidentifiedImageError
import requests, cv2
import speech_recognition as sr
from speech_recognition import AudioData
import pyttsx3
from googletrans import Translator
from pydub import AudioSegment

# --- Config ---
try:
    from config import HF_API_KEY
except Exception:
    HF_API_KEY = os.environ.get("HF_API_KEY", "")

app = Flask(__name__)
translator = Translator()

# ---------- Helpers ----------

def transcribe_audio_data(audio_bytes, rate, width):
    r = sr.Recognizer()
    audio = AudioData(audio_bytes, rate, width)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"API Error: {e}"

def generate_tts_audio(text, lang='en'):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    voices = engine.getProperty("voices")
    if lang == 'en':
        engine.setProperty("voice", voices[0].id)
    elif len(voices) > 1:
        engine.setProperty("voice", voices[1].id)
    
    tmp_file_path = "tts_output.wav"
    engine.save_to_file(text, tmp_file_path)
    engine.runAndWait()
    
    with open(tmp_file_path, "rb") as f:
        audio_bytes = f.read()
    return io.BytesIO(audio_bytes)

def post_process_image(image):
    enhancer = ImageEnhance.Brightness(image)
    img = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    return img

def analyze_sentiment(text):
    API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
    resp = requests.post(API_URL, headers=headers, json={"inputs": text})
    if resp.status_code == 200:
        js = resp.json()
        try: return js[0]['label'], js[0]['score']
        except Exception:
            try: return js[0][0]['label'], js[0][0]['score']
            except: return "ERROR", 0
    return "ERROR", 0

def apply_filter_cv(image_array, filter_type):
    if image_array is None: return None
    out = image_array.copy()
    if filter_type == "red_tint":
        out[:, :, 1] = 0
        out[:, :, 0] = 0
    elif filter_type == "green_tint":
        out[:, :, 0] = 0
        out[:, :, 2] = 0
    elif filter_type == "blue_tint":
        out[:, :, 1] = 0
        out[:, :, 2] = 0
    elif filter_type == "sobel":
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(sx, sy)
        mag = np.uint8(np.clip(mag, 0, 255))
        out = cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
    elif filter_type == "canny":
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        out = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return out

# ---------- Routes ----------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate_image", methods=["POST"])
def generate_image():
    prompt = request.form.get("prompt")
    if not prompt: return jsonify({"error": "No prompt"}), 400
    try:
        API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3-medium-diffusers"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
        resp = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=120)
        try: img = Image.open(io.BytesIO(resp.content))
        except UnidentifiedImageError: return jsonify({"error": "Model did not return image"}), 500
        img = post_process_image(img)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/sentiment", methods=["POST"])
def sentiment():
    text = request.form.get("text")
    if not text: return jsonify({"error": "No text"}), 400
    label, score = analyze_sentiment(text)
    return jsonify({"label": label, "score": float(score)})

@app.route("/filter_image", methods=["POST"])
def filter_image():
    if "file" not in request.files or "filter" not in request.form:
        return jsonify({"error": "Missing file or filter type"}), 400
    filter_type = request.form.get("filter")
    f = request.files["file"]
    data = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    out = apply_filter_cv(img, filter_type)
    ok, enc = cv2.imencode(".png", out)
    return send_file(io.BytesIO(enc.tobytes()), mimetype="image/png")

@app.route("/transcribe_audio", methods=["POST"])
def transcribe_audio():
    if "file" not in request.files: return jsonify({"error": "No audio uploaded"}), 400
    f = request.files["file"]
    content = f.read()
    if f.filename.lower().endswith(".webm"):
        try:
            seg = AudioSegment.from_file(io.BytesIO(content), format="webm")
            seg = seg.set_channels(1)
            rate = seg.frame_rate
            width = seg.sample_width
            raw = seg.raw_data
            text = transcribe_audio_data(raw, rate, width)
            return jsonify({"transcription": text})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    # WAV fallback
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        with wave.open(tmp_path, "rb") as wf:
            rate = wf.getframerate()
            width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        text = transcribe_audio_data(frames, rate, width)
        return jsonify({"transcription": text})
    finally:
        os.remove(tmp_path)

@app.route("/translate_audio", methods=["POST"])
def translate_audio():
    if "file" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    f = request.files["file"]
    audio_bytes = f.read()
    try:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        seg = seg.set_channels(1)
        rate = seg.frame_rate
        width = seg.sample_width
        raw_bytes = seg.raw_data
    except Exception:
        return jsonify({"error": "Failed to process audio"}), 500
    original_text = transcribe_audio_data(raw_bytes, rate, width)
    target_lang = request.form.get("target", "es")
    try:
        translated_text = translator.translate(original_text, dest=target_lang).text
    except Exception as e:
        translated_text = f"Translation Error: {e}"
    tts_file = generate_tts_audio(translated_text, lang=target_lang)
    audio_b64 = base64.b64encode(tts_file.read()).decode('utf-8')
    return jsonify({
        "original": original_text,
        "translated": translated_text,
        "audio_b64": audio_b64
    })

if __name__ == "__main__":
    app.run(debug=True)
