import tempfile
from atlant_coder.utils.checkpoint import _path, save, load, clear


def test_checkpoint_examples(monkeypatch):
    monkeypatch.delenv("ATLANT_CHECKPOINT_DIR", raising=False)
    """Self-check — round-trip a checkpoint through save/load/clear."""

    with tempfile.TemporaryDirectory() as td:
        # save
        snap = {"tech_task": "x", "completed_files": ["/a.py", "/b.py"], "n": 42}
        save(snap, home_dir=td, reason="test")
        assert _path(td).exists()

        # load
        loaded = load(td)
        assert loaded is not None
        assert loaded["memory"] == snap
        assert loaded["reason"] == "test"
        assert "saved_at" in loaded

        # clear
        clear(td)
        assert not _path(td).exists()

        # missing → None
        assert load(td) is None

        # corrupt → None + moved to .corrupt
        _path(td).write_text("{not json", encoding="utf-8")
        assert load(td) is None
        assert not _path(td).exists()
        corrupt = _path(td).with_suffix(_path(td).suffix + ".corrupt")
        assert corrupt.exists()
