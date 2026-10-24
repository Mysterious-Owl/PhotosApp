import io
import urllib.error

import pytest

from facegroup.adapters.models import download as dl


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_retries_then_succeeds(tmp_path, monkeypatch):
    calls = {"n": 0}

    def flaky(url, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("timed out")
        return FakeResponse(b"model-bytes")

    monkeypatch.setattr(dl.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    out = dl.download("https://example/model.onnx", tmp_path / "model.onnx")
    assert out.read_bytes() == b"model-bytes" and calls["n"] == 3
    assert not list(tmp_path.glob("*.part"))


def test_gives_a_readable_error_and_leaves_no_partial_file(tmp_path, monkeypatch):
    def down(url, timeout):
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr(dl.urllib.request, "urlopen", down)
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    with pytest.raises(dl.ModelDownloadError, match="Check the internet connection"):
        dl.download("https://example/model.onnx", tmp_path / "model.onnx")
    assert list(tmp_path.iterdir()) == []


def test_truncated_download_is_not_kept(tmp_path, monkeypatch):
    class Short(FakeResponse):
        def __init__(self):
            super().__init__(b"half")
            self.headers = {"Content-Length": "100"}

    monkeypatch.setattr(dl.urllib.request, "urlopen", lambda url, timeout: Short())
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    with pytest.raises(dl.ModelDownloadError):
        dl.download("https://example/model.onnx", tmp_path / "model.onnx", attempts=2)
    assert not (tmp_path / "model.onnx").exists()
