# agent_api_server.py (Updated to use Ollama with 'llama3:latest')

import logging
import traceback
from typing import Dict, List, Any, Callable, Optional
import os
import requests # Needed to call Ollama API

# --- Dependencies ---
# Make sure to install: pip install Flask Flask-Cors transformers torch nrclex nltk requests
# Or: pip install Flask Flask-Cors transformers tensorflow nrclex nltk requests

try:
    import nltk
    from nrclex import NRCLex
    from transformers import pipeline
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except ImportError as e:
    print(f"ERROR: Missing required libraries. Please install them.")
    print(f"pip install Flask Flask-Cors transformers torch nrclex nltk requests")
    print(f"(Or replace 'torch' with 'tensorflow' if needed)")
    print(f"Import error details: {e}")
    exit()

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- NLTK Data Download ---
def ensure_nltk_data():
    """Downloads NLTK data if not found."""
    try:
        nltk.data.find('corpora/wordnet.zip')
        nltk.data.find('corpora/omw-1.4.zip')
        logging.info("NLTK data (wordnet, omw-1.4) found.")
    except nltk.downloader.DownloadError:
        logging.warning("NLTK data not found. Attempting download...")
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            logging.info("NLTK data downloaded successfully.")
        except Exception as e:
            logging.error(f"NLTK download failed: {e}. Emotion analysis might be affected.", exc_info=True)
    except Exception as e:
         logging.error(f"An unexpected error occurred during NLTK data check: {e}", exc_info=True)

ensure_nltk_data()

# --- Load Transformers Pipeline for Analysis ---
sentiment_pipeline_analysis = None
try:
    logging.info("Loading sentiment analysis pipeline (cardiffnlp/twitter-roberta-base-sentiment)...")
    sentiment_pipeline_analysis = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        return_all_scores=True
    )
    logging.info("Sentiment analysis pipeline loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load sentiment analysis pipeline: {e}\n{traceback.format_exc()}")
    sentiment_pipeline_analysis = None

# --- ANALYSIS FUNCTION (Internal - Unchanged) ---
def analyze_text_sentiment_emotion(text_input: str) -> Dict[str, Any]:
    """
    Performs sentiment (3-label) and NRC emotion analysis.
    Returns analysis dictionary including potential 'error' key.
    """
    analysis_result: Dict[str, Any] = {
        "original_text": text_input,
        "sentiment": [],
        "dominant_sentiment": None,
        "emotions": [],
        "dominant_emotion": None,
        "error": None # Start with no error
    }
    if not text_input or not isinstance(text_input, str) or not text_input.strip():
        analysis_result["error"] = "Input text is empty or invalid."
        logging.warning(analysis_result["error"])
        return analysis_result
    # --- (Rest of the analysis logic is the same as the previous version) ---
    # 1. Sentiment Analysis
    analysis_error = None
    try:
        if sentiment_pipeline_analysis:
            sentiment_scores = sentiment_pipeline_analysis(text_input)[0]
            sentiment_data = []
            label_map = {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}
            highest_score = -1.0
            dominant_sent = None
            for item in sentiment_scores:
                label_name = label_map.get(item['label'], 'Unknown')
                score = item['score']
                sentiment_item = {"label": label_name, "score": score}
                sentiment_data.append(sentiment_item)
                if score > highest_score:
                    highest_score = score
                    dominant_sent = sentiment_item
            analysis_result["sentiment"] = sorted(sentiment_data, key=lambda x: ["Negative", "Neutral", "Positive"].index(x['label']))
            analysis_result["dominant_sentiment"] = dominant_sent
        else:
            analysis_error = "Sentiment pipeline unavailable."
            logging.warning(analysis_error)
    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}", exc_info=True)
        analysis_result["sentiment"] = []
        analysis_result["dominant_sentiment"] = None
        analysis_error = f"Sentiment analysis failed: {e}"
    # 2. Emotion Analysis (NRC Lexicon)
    try:
        nrc = NRCLex(text_input)
        emotion_freq = nrc.affect_frequencies
        emotions_data = []
        nrc_emotions_to_process = ["fear", "anger", "anticipation", "trust", "surprise", "sadness", "disgust", "joy", "positive", "negative"]
        total_score = sum(emotion_freq.get(emo, 0) for emo in nrc_emotions_to_process)
        highest_emotion_score = -1.0
        dominant_emo = None
        if total_score > 0:
            for emo in nrc_emotions_to_process:
                score = emotion_freq.get(emo, 0)
                if score > 0:
                    normalized_score = score / total_score
                    emotion_item = {"emotion": emo.capitalize(), "score": normalized_score}
                    emotions_data.append(emotion_item)
                    if normalized_score > highest_emotion_score:
                        highest_emotion_score = normalized_score
                        dominant_emo = emotion_item
            analysis_result["emotions"] = sorted(emotions_data, key=lambda x: x['emotion'])
            analysis_result["dominant_emotion"] = dominant_emo
        else:
            analysis_result["emotions"] = []
            analysis_result["dominant_emotion"] = None
    except Exception as e:
        logging.error(f"Error during emotion analysis: {e}", exc_info=True)
        analysis_result["emotions"] = []
        analysis_result["dominant_emotion"] = None
        current_error_str = analysis_error or ""
        emotion_error = f"Emotion analysis failed: {e}"
        analysis_error = f"{current_error_str}. {emotion_error}".strip().lstrip('. ')
    # Set final error state
    if analysis_error:
        analysis_result["error"] = analysis_error
    return analysis_result
    # --- (End of analysis logic) ---


# --- Function to Call Local Ollama LLM ---
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!! Assumes Ollama is running. Uses 'llama3:latest' by default now.   !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def _call_ollama_llm(prompt: str, model_name: str = "llama3:latest", ollama_url: str = "http://localhost:11434/api/generate") -> str:
    """
    Calls a local LLM through the Ollama API.

    Args:
        prompt: The prompt to send to the LLM.
        model_name: The name of the Ollama model to use (e.g., 'llama3:latest').
                    Ensure this matches a model shown by 'ollama list'.
        ollama_url: The URL of the Ollama generate endpoint.

    Returns:
        The generated text response from the LLM or an error message string.
    """
    logging.info(f"--- Calling Ollama model '{model_name}' ---")
    try:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False # Get the full response at once
        }
        # Increased timeout for potentially long LLM generation times
        response = requests.post(ollama_url, json=payload, timeout=180)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        response_data = response.json()
        # The actual generated text is in the 'response' field for non-streaming
        generated_text = response_data.get("response", "[Ollama response format error: 'response' key missing]")
        logging.info(f"--- Ollama response received (first 100 chars): {generated_text[:100]}... ---")
        return generated_text

    except requests.exceptions.ConnectionError:
        error_msg = f"[Ollama Connection Error] Could not connect to Ollama at {ollama_url}. Is Ollama running?"
        logging.error(error_msg)
        return error_msg # Return error message string
    except requests.exceptions.Timeout:
        error_msg = f"[Ollama Timeout Error] Request to Ollama timed out after 180 seconds."
        logging.error(error_msg)
        return error_msg # Return error message string
    except requests.exceptions.RequestException as e:
        # This catches HTTP errors (like 404 Not Found if model name is wrong)
        error_msg = f"[Ollama Request Error] Failed to get response from Ollama: {e}"
        logging.error(error_msg, exc_info=True)
        # Try to include Ollama's error message from the response body
        response_text = getattr(e.response, 'text', '{}') # Default to empty JSON string
        try:
            ollama_error = request.json.loads(response_text).get('error', 'Unknown error detail')
            error_msg = f"{error_msg}. Ollama error: {ollama_error}"
        except: # Keep original error if response parsing fails
             error_msg = f"{error_msg}. Response: {response_text[:200]}" # Limit length
        return error_msg # Return error message string
    except Exception as e:
        # Catch any other unexpected errors
        error_msg = f"[Ollama Unexpected Error] An error occurred: {e}"
        logging.error(error_msg, exc_info=True)
        return error_msg # Return error message string


# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Allow requests from the browser extension

# --- Agent API Endpoint (Uses Ollama) ---
@app.route("/analyze_and_rewrite", methods=["POST"])
def analyze_and_rewrite_endpoint():
    """
    API endpoint that performs analysis and agent-driven rewriting via Ollama.
    """
    if not request.is_json:
        logging.warning("Received non-JSON request")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    input_text = data.get("text", "").strip()

    if not input_text:
        logging.warning("Received request with no text")
        return jsonify({"error": "No text provided"}), 400

    # --- Step 1 & 2: Perform Analysis ---
    logging.info(f"Received text for full analysis/rewrite: '{input_text[:100]}...'")
    analysis_results = analyze_text_sentiment_emotion(input_text)

    final_response = {
        "analysis": analysis_results,
        "agent_positive_rewrite": None,
        "agent_neutral_rewrite": None,
        "error": analysis_results.get("error") # Start with potential analysis error
    }

    # --- Step 3-5: Proceed to Agent Rewrite via Ollama only if analysis was successful ---
    if final_response["error"] is None:
        try:
            original_text = analysis_results.get("original_text", input_text)
            nrc_emotions = analysis_results.get("emotions", [])

            # Format emotion details nicely for the prompt
            significant_emotions = [e for e in nrc_emotions if e.get('score', 0) > 0.1]
            if significant_emotions:
                 emotion_details = ", ".join([f"{e.get('emotion', 'N/A')} ({e.get('score', 0):.2f})" for e in significant_emotions])
            else:
                 emotion_details = "neutral or none detected above threshold"
                 logging.info("No significant emotions detected for prompt generation.")

            # Construct Prompts
            # Using f-strings with triple quotes for multi-line prompts
            positive_prompt = f"""
Instruction: Rewrite the following text to have a clearly positive sentiment.
Original Text: "{original_text}"
Detected NRC Emotions (Score > 0.1): {emotion_details}
Guidance: Maintain the core message. Enhance positive aspects or reframe negative ones informed by the detected emotions. Ensure coherence and natural language. Avoid platitudes if possible.
Positive Rewrite:"""

            neutral_prompt = f"""
Instruction: Rewrite the following text to have a strictly neutral tone.
Original Text: "{original_text}"
Detected NRC Emotions (Score > 0.1): {emotion_details}
Guidance: Remove or minimize the expression of the detected emotions (both positive and negative). Preserve essential information objectively and factually. Ensure coherence.
Neutral Rewrite:"""

            # Call Ollama LLM function (using the updated function)
            llm_func = _call_ollama_llm

            generated_positive_rewrite = llm_func(positive_prompt).strip()
            generated_neutral_rewrite = llm_func(neutral_prompt).strip()

            # Check if LLM calls returned error messages (they now start with [Ollama...)
            rewrite_errors = []
            # Store successful rewrites unless an error string was returned
            if "[Ollama" not in generated_positive_rewrite:
                final_response["agent_positive_rewrite"] = generated_positive_rewrite
            else:
                rewrite_errors.append(f"Positive rewrite failed: {generated_positive_rewrite}")

            if "[Ollama" not in generated_neutral_rewrite:
                final_response["agent_neutral_rewrite"] = generated_neutral_rewrite
            else:
                 rewrite_errors.append(f"Neutral rewrite failed: {generated_neutral_rewrite}")

            # Combine any new rewrite errors with potential analysis errors
            if rewrite_errors:
                 current_error_str = final_response["error"] or ""
                 final_response["error"] = f"{current_error_str}. {' '.join(rewrite_errors)}".strip().lstrip('. ')
            else:
                 logging.info("Agent rewrite generation via Ollama completed successfully.")

        except Exception as e:
            # Catch errors in the prompt construction / calling phase itself
            logging.error(f"Error during agent LLM rewrite step (before LLM call): {e}", exc_info=True)
            rewrite_error = f"Agent failed during rewrite setup: {e}"
            current_error_str = final_response["error"] or ""
            final_response["error"] = f"{current_error_str}. {rewrite_error}".strip().lstrip('. ')

    # --- Return combined results ---
    status_code = 500 if final_response.get("error") else 200 # Check final error status
    if status_code == 500:
         logging.error(f"Returning error response: {final_response.get('error')}")
    else:
         logging.info("Returning successful analysis and rewrite response.")

    # Ensure analysis results are always included, even if rewrites fail
    return jsonify(final_response), status_code

# --- Run Flask Server ---
if __name__ == "__main__":
    # Ensure required pipelines are loaded before starting
    if sentiment_pipeline_analysis is None:
        logging.error("Sentiment analysis pipeline failed to load. Server cannot start.")
    else:
        # Run on port 5001 to avoid conflict with potential old server on 5000
        port = 5001
        logging.info(f"Starting Agent API Server (using Ollama) on http://127.0.0.1:{port}")
        # Using threaded=True can help Flask handle multiple requests somewhat concurrently,
        # useful if the extension sends requests while the LLM is processing.
        # For production, consider a proper WSGI server like Gunicorn or Waitress.
        app.run(host="127.0.0.1", port=port, debug=False, threaded=True)