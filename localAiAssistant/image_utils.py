import requests

def analyze_image_with_llava(image_file, prompt):
    url = "http://localhost:11434/api/generate"
    files = {'image': image_file.getvalue()}
    payload = {
        "model": "llava",
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, data=payload, files=files)
        data = response.json()
        return data.get('response', str(data))
    except Exception as e:
        return f"Connection error: {e}"
