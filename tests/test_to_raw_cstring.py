"""Tests for :func:`methods.to_raw_cstring`."""

from __future__ import annotations

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from methods import to_raw_cstring


RAW_LITERAL_RE = re.compile(r'R"<!>\((.*?)\)<!>"', re.S)
MAX_LITERAL = 16 * 1024


def _decode_raw_literal(raw: str) -> list[str]:
    matches = RAW_LITERAL_RE.findall(raw)
    if not matches:
        msg = "The generated literal did not contain any raw string segments."
        raise AssertionError(msg)
    return matches


@pytest.mark.parametrize(
    "character",
    [
        "é",  # 2-byte UTF-8 sequence.
        "€",  # 3-byte UTF-8 sequence.
        "𝄞",  # 4-byte UTF-8 sequence.
    ],
)
def test_to_raw_cstring_handles_multibyte_boundaries(character: str) -> None:
    """Ensure we don't split multi-byte characters across the chunk boundary."""

    char_len = len(character.encode("utf-8"))
    prefix_len = MAX_LITERAL - char_len + 1
    test_string = "a" * prefix_len + character

    literal = to_raw_cstring(test_string)

    segments = _decode_raw_literal(literal)
    assert len(segments) >= 2  # Must split across the boundary.

    reconstructed = "".join(segments)
    assert reconstructed == test_string


def test_to_raw_cstring_does_not_emit_empty_segments() -> None:
    """Values on the chunk boundary shouldn't produce empty raw-string segments."""

    test_string = "a" * MAX_LITERAL
    literal = to_raw_cstring(test_string)

    segments = _decode_raw_literal(literal)
    assert segments == [test_string]
