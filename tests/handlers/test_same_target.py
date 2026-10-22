"""Smallest thing that fails if the wrong-file drift guard breaks."""

from atlant_coder.handlers.parallel_batch import _same_target


def test():
    # exact match
    assert _same_target("/app/main.py", "/app/main.py")
    # leading slash tolerance
    assert _same_target("app/main.py", "/app/main.py")
    assert _same_target("/app/main.py", "app/main.py")
    # whitespace tolerance
    assert _same_target("  /app/main.py  ", "/app/main.py")
    # different files → drift
    assert not _same_target("/app/main.py", "/app/other.py")
    assert not _same_target("/app/a/x.py", "/app/b/x.py")
    # empty/None → drift (never silently accept)
    assert not _same_target(None, "/app/main.py")
    assert not _same_target("", "/app/main.py")
    assert not _same_target("/app/main.py", None)
    print("ok")


if __name__ == "__main__":
    test()
