"""
test_provenance_canonical.py
-----------------------------
Deterministic canonical hashing (Phase 4 required test #13): the same
logical structure must always produce the same bytes and the same digest,
and anything that would make serialization ambiguous must be rejected.
"""

import pytest

from cv_auditor.provenance import canonical_json_bytes, sha256_of_obj
from cv_auditor.provenance.canonical import is_sha256_hex


class TestCanonicalDeterminism:
    def test_key_order_does_not_matter(self):
        a = {"b": 1, "a": {"y": 2, "x": 3}, "c": [1, 2, 3]}
        b = {"c": [1, 2, 3], "a": {"x": 3, "y": 2}, "b": 1}
        assert canonical_json_bytes(a) == canonical_json_bytes(b)
        assert sha256_of_obj(a) == sha256_of_obj(b)

    def test_repeated_hashing_is_stable(self):
        obj = {"nested": {"list": [1, 2.5, "three", None, True]}}
        digests = {sha256_of_obj(obj) for _ in range(10)}
        assert len(digests) == 1

    def test_serialization_is_compact_and_sorted(self):
        assert canonical_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'

    def test_unicode_is_not_ascii_escaped(self):
        # ensure_ascii=False -> raw UTF-8 bytes, one unambiguous encoding.
        assert canonical_json_bytes({"k": "\u00e9"}) == '{"k":"\u00e9"}'.encode("utf-8")

    def test_different_content_different_digest(self):
        assert sha256_of_obj({"a": 1}) != sha256_of_obj({"a": 2})
        assert sha256_of_obj({"a": 1}) != sha256_of_obj({"a": "1"})
        assert sha256_of_obj([1, 2]) != sha256_of_obj([2, 1])  # list order matters

    def test_digest_shape(self):
        assert is_sha256_hex(sha256_of_obj({"a": 1}))


class TestCanonicalRejections:
    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            canonical_json_bytes({"x": float("nan")})

    def test_infinity_rejected(self):
        with pytest.raises(ValueError):
            canonical_json_bytes({"x": float("inf")})

    def test_non_json_type_rejected(self):
        with pytest.raises(TypeError):
            canonical_json_bytes({"x": object()})
        with pytest.raises(TypeError):
            canonical_json_bytes({"x": {1, 2, 3}})

    def test_non_string_dict_key_rejected(self):
        with pytest.raises(TypeError):
            canonical_json_bytes({1: "a"})


class TestIsSha256Hex:
    def test_valid(self):
        assert is_sha256_hex("0" * 64)
        assert is_sha256_hex("deadbeef" * 8)

    def test_invalid(self):
        assert not is_sha256_hex("0" * 63)
        assert not is_sha256_hex("g" * 64)
        assert not is_sha256_hex(None)
        assert not is_sha256_hex(1234)
