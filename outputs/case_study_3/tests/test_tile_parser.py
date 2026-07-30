"""Tests for case_study_3/tile_parser.py."""
import pytest
from case_study_3.tile_parser import parse_tile_ids


# ---------------------------------------------------------------------------
# Happy-path: exact JSON
# ---------------------------------------------------------------------------

def test_exact_json():
    result = parse_tile_ids('{"tiles_entered": ["A1", "B1", "B2"]}')
    assert result["ok"]
    assert result["predicted_tile_sequence"] == ["A1", "B1", "B2"]
    assert result["predicted_tile_set"] == ["A1", "B1", "B2"]
    assert result["parsing_error"] is None


def test_exact_json_uppercase_normalisation():
    result = parse_tile_ids('{"tiles_entered": ["a1", "b2"]}')
    assert result["ok"]
    assert result["predicted_tile_sequence"] == ["A1", "B2"]


def test_exact_json_deduplication():
    result = parse_tile_ids('{"tiles_entered": ["A1", "B1", "A1"]}')
    assert result["ok"]
    assert result["predicted_tile_sequence"] == ["A1", "B1"]


# ---------------------------------------------------------------------------
# Markdown code fence
# ---------------------------------------------------------------------------

def test_markdown_json_fence():
    text = '```json\n{"tiles_entered": ["C3", "D4"]}\n```'
    result = parse_tile_ids(text)
    assert result["ok"]
    assert result["predicted_tile_sequence"] == ["C3", "D4"]


def test_markdown_fence_no_lang():
    text = '```\n{"tiles_entered": ["E5"]}\n```'
    result = parse_tile_ids(text)
    assert result["ok"]
    assert result["predicted_tile_sequence"] == ["E5"]


# ---------------------------------------------------------------------------
# JSON embedded in prose
# ---------------------------------------------------------------------------

def test_json_in_prose():
    text = 'Based on the image, the tiles are {"tiles_entered": ["A1", "B2"]} as shown.'
    result = parse_tile_ids(text)
    assert result["ok"]
    assert result["predicted_tile_sequence"] == ["A1", "B2"]


# ---------------------------------------------------------------------------
# Regex fallback — natural language / lists
# ---------------------------------------------------------------------------

def test_natural_language():
    text = "The path enters tiles A1, B1 and B2 from the top."
    result = parse_tile_ids(text)
    assert result["ok"]
    assert set(result["predicted_tile_set"]) == {"A1", "B1", "B2"}


def test_comma_separated():
    text = "A1, B2, C3"
    result = parse_tile_ids(text)
    assert result["ok"]
    assert result["predicted_tile_sequence"] == ["A1", "B2", "C3"]


def test_bullet_list():
    text = "- A1\n- B2\n- C3"
    result = parse_tile_ids(text)
    assert result["ok"]
    assert set(result["predicted_tile_set"]) == {"A1", "B2", "C3"}


def test_empty_tiles_list():
    result = parse_tile_ids('{"tiles_entered": []}')
    assert result["ok"]
    assert result["predicted_tile_sequence"] == []


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_empty_string():
    result = parse_tile_ids("")
    assert not result["ok"]
    assert result["parsing_error"] is not None


def test_no_tiles_found():
    result = parse_tile_ids("I cannot determine the tiles from this image.")
    assert not result["ok"]
    assert result["parsing_error"] is not None


def test_order_preserved():
    result = parse_tile_ids('{"tiles_entered": ["C3", "A1", "B2"]}')
    assert result["predicted_tile_sequence"] == ["C3", "A1", "B2"]
    # set should be sorted regardless of order
    assert result["predicted_tile_set"] == ["A1", "B2", "C3"]
