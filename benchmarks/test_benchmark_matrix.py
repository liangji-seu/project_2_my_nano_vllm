import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

import benchmark_matrix as bm


class Encoded:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    """Whitespace tokenizer with a stable, inspectable encode/decode API."""
    def encode(self, text, add_special_tokens=True):
        words = text.split()
        ids = list(range(1, len(words) + 1))
        if add_special_tokens:
            ids = [999] + ids
        return Encoded(ids)

    def decode(self, ids, skip_special_tokens=True):
        return " ".join("x" for _ in ids if not skip_special_tokens or _ != 999)


class MockHandler(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self):
        MockHandler.calls += 1
        if MockHandler.calls == 1:
            body = {"choices": [{"message": {"content": "short"}}], "usage": {"completion_tokens": 999}}
            self.send_response(200)
        elif MockHandler.calls == 2:
            body = {"error": "deliberate failure"}
            self.send_response(500)
        else:
            body = b"not json"
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else json.dumps(body).encode())

    def log_message(self, *_args):
        pass


class BenchmarkMatrixTests(unittest.TestCase):
    def test_synthetic_exact_unique_and_repeatable(self):
        tok = FakeTokenizer()
        first = bm.synthetic_records(tok, 12, 4)
        second = bm.synthetic_records(tok, 12, 4)
        self.assertEqual(first, second)
        self.assertNotEqual(first, bm.synthetic_records(tok, 12, 4, seed=9))
        self.assertEqual([x["input_tokens"] for x in first], [12] * 4)
        self.assertEqual(len({x["content"] for x in first}), 4)
        self.assertTrue(all(x["content"].startswith("request-") for x in first))
        large = bm.synthetic_records(tok, 2048, 1, seed=3)
        self.assertEqual(large[0]["input_tokens"], 2048)

    def test_sharegpt_filter_and_sample_without_replacement(self):
        tok = FakeTokenizer()
        rows = [{"content": "a b c", "source_id": "a"}, {"content": "d e f", "source_id": "b"}, {"content": "too many words here", "source_id": "c"}]
        selected = bm.select_real_records(rows, tok, target=5, count=2, tolerance=0, seed=7)
        self.assertEqual({x["source_id"] for x in selected}, {"a", "b"})
        self.assertEqual(len({x["source_id"] for x in selected}), 2)
        with self.assertRaisesRegex(ValueError, "never repeated"):
            bm.select_real_records(rows, tok, target=5, count=3, tolerance=0, seed=7)

    def test_http_success_failure_bad_json_usage_ignored(self):
        MockHandler.calls = 0
        server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            rows = [{"content": "hello", "source_id": str(i), "input_tokens": 2, "prompt_hash": str(i)} for i in range(3)]
            results, wall = bm.run_requests(f"http://127.0.0.1:{server.server_port}", "fake", rows, 8, 1, 3, FakeTokenizer())
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
        self.assertGreater(wall, 0)
        self.assertEqual([x["status"] for x in results].count("success"), 1)
        success = next(x for x in results if x["status"] == "success")
        self.assertEqual(success["estimated_output_tokens"], 1)
        self.assertEqual(success["response_chars"], 5)
        self.assertIsNotNone(success["response_text_sha256"])
        self.assertEqual(len([x for x in results if x["error"]]), 2)
        self.assertTrue(all(x["latency_ms"] is not None for x in results))

    def test_output_files_schema(self):
        with TemporaryDirectory() as tmp:
            args = type("Args", (), {"output_dir": tmp})()
            out = bm.write_outputs(args, {"status": "dry_run"}, [{"repeat": 0, "input_target": 4, "output_budget": 2, "concurrency": 1, "requested": 1, "valid": None}], [], {"case_count": 1, "total_requests": 1, "manifest": []})
            self.assertEqual({p.name for p in Path(out).iterdir()}, {"summary.json", "summary.csv"})
            summary = json.loads((Path(out) / "summary.json").read_text())
            self.assertEqual(summary["planned"]["total_requests"], 1)
            self.assertEqual(summary["metadata"]["status"], "dry_run")


if __name__ == "__main__":
    unittest.main()
