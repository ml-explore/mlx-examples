import requests
import json

def process_sse_stream(url, headers, data):
    response = requests.post(url, headers=headers, json=data, stream=True)

    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        print(response.text)
        return

    full_content = ""

    try:
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    event_data = line[6:]  # Remove 'data: ' prefix
                    if event_data == '[DONE]':
                        print("\nStream finished. ✅")
                        break
                    try:
                        chunk_data = json.loads(event_data)
                        content = chunk_data['choices'][0]['delta']['content']
                        full_content += content
                        print(content, end='', flush=True)
                    except json.JSONDecodeError:
                        print(f"\nFailed to decode JSON: {event_data}")
                    except KeyError:
                        print(f"\nUnexpected data structure: {chunk_data}")

    except KeyboardInterrupt:
        print("\nStream interrupted by user.")
    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
        "messages": [{"role": "user", "content": "Hi, how are you?"}],
        "max_tokens": 500,
        "stream": True
    }
    process_sse_stream(url, headers, data)
