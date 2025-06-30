import base64
import requests


def analyze_image_with_llava(image_file, prompt, history=None):
    url = "http://localhost:11434/api/chat"

    # Convert image to base64
    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    # Build the message history
    messages = []

    # Add previous chat history if available
    if history:
        for role, content in history:
            if isinstance(content, list):  # For content with image+text in history
                messages.append({"role": role, "content": content})
            else:
                messages.append({"role": role, "content": content})

    # Append current user message with image
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
        ]
    })

    payload = {
        "model": "llava",
        "messages": messages,
        "stream": False
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        data = response.json()
        reply = data.get('message', {}).get('content', 'No response.')
        return reply
    else:
        return f"❌ Error: {response.text}"
