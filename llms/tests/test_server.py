import http
import threading
import unittest

import requests
from mlx_lm.server import APIHandler
from mlx_lm.utils import load


class DummyModelProvider:
    def __init__(self):
        HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        self.model, self.tokenizer = load(HF_MODEL_PATH)

    def load(self, model):
        assert model in ["default_model", "chat_model"]
        return self.model, self.tokenizer


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_provider = DummyModelProvider()
        cls.server_address = ("localhost", 0)
        cls.httpd = http.server.HTTPServer(
            cls.server_address,
            lambda *args, **kwargs: APIHandler(cls.model_provider, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()

    def test_handle_completions(self):
        url = f"http://localhost:{self.port}/v1/completions"

        post_data = {
            "model": "default_model",
            "prompt": "Once upon a time",
            "max_tokens": 10,
            "temperature": 0.5,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "repetition_context_size": 20,
            "stop": "stop sequence",
        }

        response = requests.post(url, json=post_data)

        response_body = response.text

        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_chat_completions(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)


if __name__ == "__main__":
    unittest.main()
