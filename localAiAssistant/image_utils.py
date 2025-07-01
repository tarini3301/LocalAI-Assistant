import base64
import requests


def analyze_image_with_llava(image_file, prompt, history=None):
    """
    Analyze an image with a text prompt using LLaVA via Ollama's API.

    Args:
        image_file: Uploaded image file (from Streamlit).
        prompt: User's text question about the image.
        history: List of (role, content) tuples for chat history.

    Returns:
        Model response string.
    """

    url = "http://localhost:11434/api/generate"  # ✅ Correct endpoint

    try:
        image_file.seek(0)  # Ensure reading from start
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        # Build chat prompt with history
        full_prompt = ""
        if history:
            for role, content in history:
                full_prompt += f"{role}: {content}\n"
        full_prompt += f"User: {prompt}\nAssistant:"

        # Payload for Ollama
        payload = {
            "model": "llava",   # ✅ Make sure this model is pulled
            "prompt": full_prompt,
            "images": [image_base64],
            "stream": False
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        reply = data.get('response', 'No response from model.').strip()

        return reply

    except requests.exceptions.HTTPError as http_err:
        return f"❌ HTTP error: {http_err} - {response.text}"
    except requests.exceptions.ConnectionError:
        return "❌ Connection error. Is Ollama running at http://localhost:11434?"
    except requests.exceptions.Timeout:
        return "❌ Request timed out."
    except Exception as err:
        return f"❌ An error occurred: {err}"
