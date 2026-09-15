from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    page: int
    index: int
    text: str


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
