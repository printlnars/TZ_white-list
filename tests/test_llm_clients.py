"""HTTP-клиенты проверяем на локальном фейковом сервере: формат запроса и разбор ответа."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from docextract.llm import OllamaClient, OpenAICompatClient
from docextract.pipeline import extract

ANSWER = json.dumps({"doc_type": "invoice", "supplier_name": None, "supplier_bin": None, "doc_number": "1245",
                     "doc_date": None, "currency": None, "total_amount": None, "vat_amount": None,
                     "line_items": [], "warnings": []})


@pytest.fixture
def server():
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body, dict(self.headers)))
            if self.path == "/api/chat":
                out = {"message": {"role": "assistant", "content": ANSWER}}
            else:
                out = {"choices": [{"message": {"role": "assistant", "content": ANSWER}}]}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", seen
    srv.shutdown()


def test_ollama_request_uses_json_schema_and_temperature_0(server):
    url, seen = server
    r = extract("Счёт № 1245", OllamaClient(model="qwen2.5:7b", host=url))
    assert r.ok and r.extraction.doc_number == "1245"
    path, body, _ = seen[0]
    assert path == "/api/chat"
    assert body["model"] == "qwen2.5:7b" and body["stream"] is False
    assert body["options"]["temperature"] == 0
    assert "supplier_bin" in body["format"]["properties"]  # constrained decoding по схеме


def test_openai_compat_request(server):
    url, seen = server
    r = extract("Счёт № 1245", OpenAICompatClient(model="m", base_url=url, api_key="k"))
    assert r.ok
    path, body, headers = seen[0]
    assert path == "/chat/completions" and body["response_format"] == {"type": "json_object"}
    assert headers["Authorization"] == "Bearer k"
