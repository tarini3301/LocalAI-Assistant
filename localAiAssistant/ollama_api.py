import requests

def chat_with_model(prompt, model=None, context="chat"):
    url = "http://localhost:11434/api/generate"

    # Model selection logic
    if model is None:
        if context == "chat":
            model = "llama3"
        elif context == "fast_chat":
            model = "phi3"
        elif context == "code":
            model = "deepseek-coder"
        elif context == "document":
            model = "mistral"
        elif context == "creative":
            model = "llama3"
        else:
            model = "llama3"  # Default fallback

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 2048,
            "num_gpu_layers": 35,
            "temperature": 0.7 if context != "creative" else 1.0,
            "top_p": 0.9,
            "top_k": 50
        }
    }


    try:
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            data = response.json()
            if 'response' in data:
                return data['response'].strip()
            else:
                return str(data)
        else:
            return f"Error {response.status_code}: {response.text}"

    except Exception as e:
        return f"Failed to connect to Ollama: {e}"
