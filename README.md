
# Machine Translation - Flask + Docker

Minimal Flask web app that serves a sequence-to-sequence translation model.
Use your *own trained model* by copying its exported tokenizer+weights into `model/`.

## Run locally (without Docker)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Use your local model (put it under ./model), or use a HF model id as fallback:
export MODEL_ID=Helsinki-NLP/opus-mt-en-fr
python app.py
# open http://127.0.0.1:8000
```

## Run with Docker
```bash
docker build -t mt-app .
# Option A: use a HF Hub model ID at runtime
docker run -p 8000:8000 -e MODEL_ID=Helsinki-NLP/opus-mt-en-fr mt-app
# Option B: copy your local exported model into ./model before building, then:
docker run -p 8000:8000 mt-app
```

## Test the JSON API
```bash
curl -X POST http://localhost:8000/translate -H "Content-Type: application/json" -d '{"text":"Hello world"}'
```

## Exporting your own model from Colab
In your Colab notebook, after training:
```python
from transformers import AutoTokenizer
tokenizer.save_pretrained("/content/model")
model.save_pretrained("/content/model")
# Then:  download the /content/model folder as a zip and place it under this project's ./model
```

## Notes
- Default port is 8000; configure with `PORT` env var.
- You can control max decode length via `MAX_LENGTH` env var.
