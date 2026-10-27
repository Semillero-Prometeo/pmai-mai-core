from __future__ import annotations

from pmai_core.resources.audio.wake_match import WakeConfirm, match_wake_phrase

PHRASES = ["hey fede", "fede", "federico"]


def test_exact_phrase_matches() -> None:
    assert match_wake_phrase("hey fede", PHRASES, True) == "hey fede"
    assert match_wake_phrase("fede", PHRASES, True) == "fede"
    assert match_wake_phrase("federico", PHRASES, True) == "federico"


def test_suffix_matches_longest_phrase() -> None:
    assert match_wake_phrase("oye hey fede", PHRASES, True) == "hey fede"


def test_substring_inside_another_word_does_not_match() -> None:
    assert match_wake_phrase("confederación", PHRASES, True) is None
    assert match_wake_phrase("federico", ["fede"], True) is None


def test_partial_match_false_rejects_suffix() -> None:
    assert match_wake_phrase("oye hey fede", PHRASES, False) is None
    assert match_wake_phrase("hey fede", PHRASES, False) == "hey fede"


def test_punctuation_and_case_are_ignored() -> None:
    assert match_wake_phrase("Hey, Fede!", PHRASES, False) == "hey fede"


def test_one_partial_does_not_open() -> None:
    confirm = WakeConfirm(2)
    assert confirm.observe("hey fede", False) is False


def test_two_partials_open() -> None:
    confirm = WakeConfirm(2)
    assert confirm.observe("hey fede", False) is False
    assert confirm.observe("hey fede", False) is True


def test_different_partial_resets() -> None:
    confirm = WakeConfirm(2)
    assert confirm.observe("hey fede", False) is False
    assert confirm.observe("fede", False) is False
    assert confirm.observe("fede", False) is True


def test_final_opens_once() -> None:
    confirm = WakeConfirm(2)
    assert confirm.observe("hey fede", True) is True


def test_empty_final_does_not_open() -> None:
    confirm = WakeConfirm(2)
    assert confirm.observe(None, True) is False


def test_silence_resets_partial_count() -> None:
    confirm = WakeConfirm(2)
    assert confirm.observe("hey fede", False) is False
    confirm.reset()
    assert confirm.observe("hey fede", False) is False
