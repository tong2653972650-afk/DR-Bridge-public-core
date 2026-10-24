import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "release_verifier", Path(__file__).resolve().parents[1]/"scripts/verify_release.py"
)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)
MANIFEST_SPEC = importlib.util.spec_from_file_location(
    "release_manifest", Path(__file__).resolve().parents[1]/"scripts/build_manifest.py"
)
MANIFEST = importlib.util.module_from_spec(MANIFEST_SPEC)
MANIFEST_SPEC.loader.exec_module(MANIFEST)


class ReleaseIntegrityTests(unittest.TestCase):
    def test_layout_and_comments_do_not_change_method_fingerprint(self):
        first = ast.parse("def f(x):\n    return x + 1\n").body[0]
        second = ast.parse("# heading\ndef f( x ):\n    # detail\n    return (x+1)\n").body[0]
        self.assertEqual(VERIFIER.ast_digest(first), VERIFIER.ast_digest(second))

    def test_signature_change_changes_method_fingerprint(self):
        first = ast.parse("def f(x):\n    return x + 1\n").body[0]
        second = ast.parse("def f(x, region_bias=None):\n    return x + 1\n").body[0]
        self.assertNotEqual(VERIFIER.ast_digest(first), VERIFIER.ast_digest(second))

    def test_body_change_changes_method_fingerprint(self):
        first = ast.parse("def f(x):\n    return x + 1\n").body[0]
        second = ast.parse("def f(x):\n    return x + 2\n").body[0]
        self.assertNotEqual(VERIFIER.ast_digest(first), VERIFIER.ast_digest(second))

    def test_empty_type_parameters_are_version_neutral(self):
        before = ast.parse("def f(x):\n    return x\n").body[0]
        after = ast.parse("def f(x):\n    return x\n").body[0]
        after.type_params = []
        if hasattr(before, "type_params"):
            del before.type_params
        self.assertEqual(VERIFIER.ast_digest(before), VERIFIER.ast_digest(after))

    def test_special_literal_fields_are_serializable(self):
        node = ast.parse("def f():\n    return b'x', 1j, ..., None\n").body[0]
        self.assertEqual(len(VERIFIER.ast_digest(node)), 64)

    def test_generated_metadata_is_not_release_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("README.md", ".git/HEAD", ".venv/marker", "src/demo.egg-info/PKG-INFO",
                         "src/drbridge/__pycache__/model.pyc", "outputs/result.json"):
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("test", encoding="utf-8")
            names = {p.relative_to(root).as_posix() for p in VERIFIER.payload_files(root)}
            self.assertEqual(names, {"README.md"})

    def test_unexpected_source_is_counted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"unexpected.py").write_text("x = 1\n", encoding="utf-8")
            self.assertEqual([p.name for p in VERIFIER.payload_files(root)], ["unexpected.py"])


class ManifestLineEndingTests(unittest.TestCase):
    def test_manifest_is_written_and_checked_with_lf(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"README.md").write_bytes(b"Example\n")
            report = MANIFEST.build_manifest(root)
            self.assertEqual(report["files_hashed"], 1)
            self.assertNotIn(b"\r", (root/"MANIFEST.sha256").read_bytes())
            self.assertEqual(MANIFEST.build_manifest(root, check=True)["manifest"], "matched")

    def test_crlf_and_cr_only_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for content in (b"Example\r\n", b"Example\r"):
                with self.subTest(content=content):
                    (root/"README.md").write_bytes(content)
                    with self.assertRaisesRegex(ValueError, "LF line endings"):
                        MANIFEST.build_manifest(root)

    def test_utf8_bom_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"example.json").write_bytes(b"\xef\xbb\xbf{}\n")
            with self.assertRaisesRegex(ValueError, "BOM"):
                MANIFEST.build_manifest(root)

    def test_stale_manifest_is_rejected_without_rewriting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            readme = root/"README.md"
            readme.write_bytes(b"Before\n")
            MANIFEST.build_manifest(root)
            previous = (root/"MANIFEST.sha256").read_bytes()
            readme.write_bytes(b"After\n")
            with self.assertRaisesRegex(ValueError, "stale"):
                MANIFEST.build_manifest(root, check=True)
            self.assertEqual((root/"MANIFEST.sha256").read_bytes(), previous)

    def test_binary_files_are_hashed_without_text_conversion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = b"\x89PNG\r\n\x00\xff"
            (root/"fixture.png").write_bytes(original)
            self.assertEqual(MANIFEST.build_manifest(root)["files_hashed"], 1)
            self.assertEqual((root/"fixture.png").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
