import tempfile
import pathlib
from atlant_coder.utils.smoke_check import smoke_check


def test_smoke_check_examples(monkeypatch):
    monkeypatch.delenv("ATLANT_CHECKPOINT_DIR", raising=False)
    """Self-check — assert-based, no framework."""

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)

        # clean file, valid stdlib import
        good = root / "good.py"
        good.write_text("import os\nimport json\nx = 1\n")
        assert smoke_check(str(good), td) == [], "clean file must pass"

        # sibling import that exists in project
        (root / "helper.py").write_text("Y = 2\n")
        sib = root / "uses_helper.py"
        sib.write_text("from helper import Y\nprint(Y)\n")
        assert smoke_check(str(sib), td) == [], "sibling import must resolve"

        # missing import
        bad_import = root / "bad_import.py"
        bad_import.write_text("import definitely_not_a_real_module_xyz\n")
        errs = smoke_check(str(bad_import), td)
        assert errs and "definitely_not_a_real_module_xyz" in errs[0], errs

        # broken syntax
        bad_synth = root / "bad_synth.py"
        bad_synth.write_text("def broken(:\n")
        errs = smoke_check(str(bad_synth), td)
        assert errs and "py_compile FAILED" in errs[0], errs

        # relative imports get a pass (not our job to validate package layout)
        rel = root / "rel.py"
        rel.write_text("from . import whatever\n")
        # py_compile actually rejects module-level relative imports outside a package,
        # so this is expected to produce a py_compile error, NOT an import-miss.
        errs = smoke_check(str(rel), td)
        assert not errs or "py_compile" in errs[0], errs

        # non-.py file → skipped
        (root / "readme.md").write_text("# hi\n")
        assert smoke_check(str(root / "readme.md"), td) == []

        # NEW: sibling-drift — importing a name the target doesn't define
        (root / "models.py").write_text(
            "class Foo:\n    pass\n\nclass Bar:\n    pass\n"
        )
        drift = root / "uses_drift.py"
        drift.write_text("from models import Foo, GhostRow\nprint(Foo)\n")
        errs = smoke_check(str(drift), td)
        assert errs and "GhostRow" in errs[-1], errs
        assert "Foo" not in errs[-1], "Foo exists, should not be listed"

        # NEW: relative import name-check
        pkg = root / "mypkg"
        pkg.mkdir(exist_ok=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "sib.py").write_text("class Real:\n    pass\n")
        rel_ok = pkg / "uses_sib.py"
        rel_ok.write_text("from .sib import Real\n")
        assert (
            smoke_check(str(rel_ok), td) == []
        ), "relative import of existing name must pass"

        rel_bad = pkg / "uses_missing.py"
        rel_bad.write_text("from .sib import Missing\n")
        errs = smoke_check(str(rel_bad), td)
        assert errs and "Missing" in errs[-1], errs
