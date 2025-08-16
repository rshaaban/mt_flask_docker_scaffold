# app_keras.py
import os, pickle, numpy as np, traceback
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

APP = Flask(__name__, template_folder="templates", static_folder="static")

MODEL_DIR = os.environ.get("MODEL_DIR", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "translation_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "eng_tokenizer.pkl")

translation_model = None
encoder_model = None
decoder_model = None
tokenizer = None
index_word = None
word_index = None
max_sequence_length = None
lstm_units = None

def safe_print(*a, **k):
    try:
        print(*a, **k)
    except:
        pass

def try_load_all():
    """Load model + tokenizer and build encoder/decoder inference models robustly."""
    global translation_model, encoder_model, decoder_model, tokenizer
    global index_word, word_index, max_sequence_length, lstm_units

    if translation_model is not None and encoder_model is not None and decoder_model is not None:
        return True

    # Check files
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        safe_print("Model or tokenizer not found in", MODEL_DIR)
        return False

    # Load model & tokenizer
    safe_print("Loading Keras model. This may take a moment...")
    translation_model = load_model(MODEL_PATH, compile=False)
    safe_print("Loaded model:", translation_model.name)

    safe_print("Loading tokenizer...")
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    # Build word_index / index_word mappings
    if hasattr(tokenizer, "word_index"):
        word_index = tokenizer.word_index
        index_word = getattr(tokenizer, "index_word", None)
        if index_word is None:
            # build inverse
            index_word = {v:k for k,v in word_index.items()}
    elif isinstance(tokenizer, dict):
        # token dict (either id->word or word->id)
        sample_keys = list(tokenizer.keys())
        if sample_keys and isinstance(sample_keys[0], int):
            index_word = {int(k): v for k, v in tokenizer.items()}
            word_index = {v:int(k) for k,v in tokenizer.items()}
        else:
            word_index = {k:int(v) for k,v in tokenizer.items()}
            index_word = {v:k for k,v in word_index.items()}
    else:
        safe_print("Tokenizer type unknown:", type(tokenizer))

    # Try to infer max sequence length from model input shapes
    try:
        shapes = translation_model.input_shape
        # shapes might be a list/tuple for multiple inputs
        if isinstance(shapes, (list, tuple)):
            # take encoder input (first)
            max_sequence_length = int(shapes[0][1]) if shapes[0] and len(shapes[0])>1 else None
        else:
            max_sequence_length = int(shapes[1])
    except Exception:
        max_sequence_length = None

    safe_print("Inferred max_sequence_length:", max_sequence_length)

    # Find LSTM and Embedding and Dense layers
    lstm_layers = [l for l in translation_model.layers if l.__class__.__name__ == "LSTM" or isinstance(l, tf.keras.layers.LSTM)]
    embed_layers = [l for l in translation_model.layers if l.__class__.__name__ == "Embedding" or isinstance(l, tf.keras.layers.Embedding)]
    dense_layers = [l for l in translation_model.layers if l.__class__.__name__ == "Dense" or isinstance(l, tf.keras.layers.Dense)]

    safe_print("Found LSTM layers:", [l.name for l in lstm_layers])
    safe_print("Found Embedding layers:", [l.name for l in embed_layers])
    safe_print("Found Dense layers:", [l.name for l in dense_layers])

    # Use model.inputs for encoder/decoder input tensors (should be two)
    try:
        encoder_input_tensor = translation_model.inputs[0]
        decoder_input_tensor = translation_model.inputs[1]
        safe_print("Found model.inputs[0] and [1]. Shapes:", encoder_input_tensor.shape, decoder_input_tensor.shape)
        if max_sequence_length is None:
            try:
                max_sequence_length = int(encoder_input_tensor.shape[1])
            except Exception:
                max_sequence_length = 20
    except Exception:
        safe_print("Could not find translation_model.inputs[] as expected.")
        return False

    # Decide LSTM layers: assume first LSTM is encoder, second is decoder (common pattern)
    if len(lstm_layers) < 2:
        safe_print("Not enough LSTM layers found. Model may be custom.")
        return False

    encoder_lstm_layer = lstm_layers[0]
    decoder_lstm_layer = lstm_layers[1]

    # Try to extract encoder states (the LSTM layer output may be a list)
    enc_out = encoder_lstm_layer.output
    if isinstance(enc_out, (list, tuple)):
        # expected: [output, state_h, state_c]
        try:
            state_h_enc = enc_out[1]
            state_c_enc = enc_out[2]
        except Exception:
            safe_print("Encoder LSTM outputs unexpected format.")
            return False
    else:
        # Unexpected: maybe return_state was False at save time
        safe_print("Encoder LSTM layer.output is a single tensor; cannot retrieve states.")
        return False

    # Build encoder_model
    try:
        encoder_model = Model(encoder_input_tensor, [state_h_enc, state_c_enc])
        safe_print("Built encoder_model.")
    except Exception as e:
        safe_print("Failed to build encoder_model:", e)
        safe_print(traceback.format_exc())
        return False

    # Determine LSTM units from the state tensor shape
    try:
        lstm_units = int(state_h_enc.shape[-1])
        safe_print("Inferred LSTM units:", lstm_units)
    except Exception:
        lstm_units = None

    # Build decoder inference model
    try:
        # new inputs for decoder states at inference
        dec_state_input_h = Input(shape=(lstm_units,), name="dec_state_input_h")
        dec_state_input_c = Input(shape=(lstm_units,), name="dec_state_input_c")

        # get decoder embedding (if present)
        if embed_layers:
            # assume the second embedding is decoder embedding (encoder=first, decoder=second)
            if len(embed_layers) >= 2:
                decoder_embedding_layer = embed_layers[1]
            else:
                decoder_embedding_layer = embed_layers[0]
            decoder_embedded = decoder_embedding_layer(decoder_input_tensor)
        else:
            decoder_embedded = decoder_input_tensor  # fallback

        # call decoder LSTM with the new state inputs
        decoder_lstm = decoder_lstm_layer
        decoder_outputs_and_states = decoder_lstm(decoder_embedded, initial_state=[dec_state_input_h, dec_state_input_c])
        # decoder_lstm when return_state=True returns (outputs, state_h, state_c)
        if isinstance(decoder_outputs_and_states, (list, tuple)):
            decoder_outputs = decoder_outputs_and_states[0]
            state_h_dec = decoder_outputs_and_states[1]
            state_c_dec = decoder_outputs_and_states[2]
        else:
            safe_print("Decoder LSTM returned unexpected structure.")
            return False

        # find decoder dense (softmax) layer: prefer last Dense layer
        if dense_layers:
            decoder_dense_layer = dense_layers[-1]
        else:
            safe_print("No Dense layer found for output projection.")
            return False

        decoder_outputs = decoder_dense_layer(decoder_outputs)

        decoder_model = Model(
            [decoder_input_tensor, dec_state_input_h, dec_state_input_c],
            [decoder_outputs, state_h_dec, state_c_dec]
        )
        safe_print("Built decoder_model.")
    except Exception as e:
        safe_print("Failed to build decoder_model:", e)
        safe_print(traceback.format_exc())
        return False

    # set globals
    globals().update({
        "translation_model": translation_model,
        "encoder_model": encoder_model,
        "decoder_model": decoder_model,
        "tokenizer": tokenizer,
        "word_index": word_index,
        "index_word": index_word,
        "max_sequence_length": max_sequence_length,
        "lstm_units": lstm_units
    })

    safe_print("All inference models built successfully.")
    return True


def find_start_token_index():
    """Try common start tokens in tokenizer; fallback to 1."""
    candidates = ["<start>", "<s>", "startseq", "sos", "<sos>", "▁start", "[START]"]
    for t in candidates:
        if word_index and t in word_index:
            return word_index[t]
    # try common numeric tokens
    for fallback in [1, 2, 3]:
        if index_word and fallback in index_word:
            return fallback
    return 1


def find_end_tokens_set():
    c = {"<end>", "</s>", "endseq", "eos", "<eos>", "[END]"}
    # Map to any ids that exist in word_index
    ids = set()
    for w in c:
        if word_index and w in word_index:
            ids.add(word_index[w])
    return ids


def decode_sequence(input_seq):
    """Greedy decoding using encoder_model and decoder_model."""
    if encoder_model is None or decoder_model is None:
        raise RuntimeError("Inference models not loaded.")

    # Encode input sequence to get initial states
    states_value = encoder_model.predict(input_seq)

    # Prepare target_seq with start token
    start_index = find_start_token_index()
    target_seq = np.array([[start_index]])

    stop_ids = find_end_tokens_set()
    stop_condition = False
    decoded_tokens = []
    max_len = max_sequence_length or 40

    steps = 0
    repeat_count = 0
    last_word = None

    while not stop_condition and steps < (max_len + 5):
        # predict next token
        output_tokens, h, c = decoder_model.predict([target_seq] + list(states_value))
        if output_tokens.ndim == 3:
            sampled_token_index = int(np.argmax(output_tokens[0, -1, :]))
        else:
            sampled_token_index = int(np.argmax(output_tokens[0]))

        # stop if it's an end token or padding
        if sampled_token_index == 0 or sampled_token_index in stop_ids:
            break

        sampled_word = index_word.get(sampled_token_index, "")
        if sampled_word:
            # check repetition
            if sampled_word == last_word:
                repeat_count += 1
            else:
                repeat_count = 0
            last_word = sampled_word

            if repeat_count >= 2:  # stop after 3 repeats
                break

            decoded_tokens.append(sampled_word)

        # Update target_seq and states
        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]
        steps += 1

    return " ".join(decoded_tokens)


def translate_text(text):
    if not try_load_all():
        raise RuntimeError("Model/tokenizer not available or failed to build inference models.")

    # convert text -> sequence
    if hasattr(tokenizer, "texts_to_sequences"):
        seq = tokenizer.texts_to_sequences([text])
    else:
        # naive split mapping
        seq = [[word_index.get(w, 1) for w in text.lower().split()]]

    # pad
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = pad_sequences(seq, maxlen=(max_sequence_length or 20), padding="post")
    return decode_sequence(seq)


@APP.route("/", methods=["GET"])
def index():
    try_load_all()
    return render_template("index.html")


@APP.route("/health", methods=["GET"])
def health():
    ok = try_load_all()
    if ok:
        return jsonify({"ok": True, "model": str(translation_model.name), "max_seq_len": max_sequence_length})
    return jsonify({"ok": False, "error": "Model/tokenizer not ready. Check server logs."}), 500


@APP.route("/translate", methods=["POST"])
def translate():
    if request.is_json:
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
    else:
        text = (request.form.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Text is required."}), 400

    try:
        result = translate_text(text)
        return jsonify({"translation": result})
    except Exception as e:
        safe_print("Translation error:", e)
        safe_print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
