
import os
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

app = Flask(__name__)

# Configuration
MODEL_DIR = os.environ.get("MODEL_DIR", "model")
MODEL_ID = os.environ.get("MODEL_ID", "")  # e.g. "Helsinki-NLP/opus-mt-en-fr" as a fallback
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "256"))

_translator = None
_model_used = None

def load_translator():
    global _translator, _model_used
    if _translator is not None:
        return _translator

    # Decide where to load the model from
    model_path = None
    # Prefer a local model if present (must contain a config.json)
    if os.path.isdir(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        model_path = MODEL_DIR
    elif MODEL_ID:
        model_path = MODEL_ID
    else:
        # Safe default to let the app work even before the user's own model is copied in
        model_path = "Helsinki-NLP/opus-mt-en-fr"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    _translator = pipeline("translation", model=model, tokenizer=tokenizer, max_length=MAX_LENGTH)
    _model_used = model_path
    return _translator

@app.route("/", methods=["GET"])
def index():
    # Lazy-load so first request triggers model load
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    try:
        tr = load_translator()
        return jsonify({"ok": True, "model": _model_used}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/translate", methods=["POST"])
def translate():
    # Support both form submission and JSON API
    text = ""
    if request.is_json:
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
    else:
        text = (request.form.get("text") or "").strip()

    if not text:
        error_msg = "Text is required."
        if request.is_json:
            return jsonify({"error": error_msg}), 400
        return render_template("index.html", input_text=text, result="", error=error_msg), 400

    translator = load_translator()
    try:
        out = translator(text)
        translation = out[0]["translation_text"]
    except Exception as e:
        if request.is_json:
            return jsonify({"error": f"Translation failed: {e}"}), 500
        return render_template("index.html", input_text=text, result="", error=f"Translation failed: {e}"), 500

    if request.is_json:
        return jsonify({"translation": translation, "model": _model_used})
    return render_template("index.html", input_text=text, result=translation, error="")

if __name__ == "__main__":
    # Dev server (use gunicorn in Docker/production)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=True)
