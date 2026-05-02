import os
import subprocess
import sys
import tempfile
import textwrap


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from minisolver.MiniModel import OptimalControlModel


def require(text, needle):
    if needle not in text:
        raise AssertionError(f"missing generated snippet: {needle}")


def reject(text, needle):
    if needle in text:
        raise AssertionError(f"unexpected generated snippet: {needle}")


def expect_value_error(fn, needle):
    try:
        fn()
    except ValueError as exc:
        if needle not in str(exc):
            raise AssertionError(f"expected error containing {needle!r}, got {exc!r}") from exc
        return
    raise AssertionError("expected ValueError")


def compile_and_run(tmpdir, source_name, exe_name, source):
    source_path = os.path.join(tmpdir, source_name)
    exe_path = os.path.join(tmpdir, exe_name)
    with open(source_path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(source))
    subprocess.run(
        [
            "g++", "-std=c++17", "-DUSE_CUSTOM_MATRIX",
            f"-I{ROOT}/include", f"-I{tmpdir}", source_path, "-o", exe_path
        ],
        check=True,
    )
    subprocess.run([exe_path], check=True)


def generate_header_text(model, header_name, **generate_kwargs):
    with tempfile.TemporaryDirectory() as tmpdir:
        model.generate(tmpdir, **generate_kwargs)
        header_path = os.path.join(tmpdir, header_name)
        with open(header_path, "r", encoding="utf-8") as f:
            return f.read()
