from __future__ import annotations

import re

_DROP = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_words(text: str) -> list[str]:
    cleaned = _DROP.sub(" ", text.lower()).replace("_", " ")
    return [word for word in cleaned.split() if word]


def match_wake_phrase(text: str, phrases: list[str], partial_match: bool) -> str | None:
    words = normalize_words(text)
    if not words:
        return None
    ranked = sorted(
        ((normalize_words(phrase), phrase) for phrase in phrases),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for phrase_words, _original in ranked:
        if not phrase_words:
            continue
        if words == phrase_words:
            return " ".join(phrase_words)
        if (
            partial_match
            and len(words) >= len(phrase_words)
            and words[-len(phrase_words) :] == phrase_words
        ):
            return " ".join(phrase_words)
    return None


class WakeConfirm:
    def __init__(self, chunks: int) -> None:
        self._need = max(1, chunks)
        self._phrase: str | None = None
        self._count = 0

    def reset(self) -> None:
        self._phrase = None
        self._count = 0

    def observe(self, phrase: str | None, is_final: bool) -> bool:
        if is_final:
            self.reset()
            return phrase is not None
        if phrase is None:
            self.reset()
            return False
        if phrase == self._phrase:
            self._count += 1
        else:
            self._phrase = phrase
            self._count = 1
        if self._count >= self._need:
            self.reset()
            return True
        return False
