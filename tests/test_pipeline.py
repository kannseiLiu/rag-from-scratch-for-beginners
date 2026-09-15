import pytest

from beginner_pdf_rag.models import Page


class KeywordEmbedder:
    def embed(self, texts):
        return [[float("alpha" in text), float("beta" in text)] for text in texts]


class RecordingGenerator:
    def __init__(self, answer):
        self.answer = answer
        self.context = None

    def generate(self, question, context):
        self.context = context
        return self.answer


class FakeEmbedder:
    def embed(self, texts):
        return [[1.0] for _ in texts]


class FakeGenerator:
    def generate(self, question, context):
        return "answer"


def test_pipeline_indexes_retrieves_and_builds_page_context(monkeypatch):
    from beginner_pdf_rag.pipeline import PdfRag

    monkeypatch.setattr(
        "beginner_pdf_rag.pipeline.load_pdf", lambda _: [Page(4, "alpha beta")]
    )
    embedder = KeywordEmbedder()
    generator = RecordingGenerator("Grounded answer [Page 4]")
    rag = PdfRag(embedder, generator, chunk_size=20, overlap=2)

    assert rag.index("paper.pdf") == 1
    answer = rag.ask("alpha", top_k=1)

    assert answer.text == "Grounded answer [Page 4]"
    assert generator.context.startswith("[Page 4 | score=")


def test_answer_sources_are_the_exact_ranked_results(monkeypatch):
    from beginner_pdf_rag.pipeline import PdfRag

    monkeypatch.setattr(
        "beginner_pdf_rag.pipeline.load_pdf",
        lambda _: [Page(2, "alpha"), Page(5, "beta")],
    )
    rag = PdfRag(
        KeywordEmbedder(), RecordingGenerator("answer"), chunk_size=20, overlap=2
    )
    rag.index("paper.pdf")

    answer = rag.ask("beta", top_k=1)

    assert [(item.chunk.page, item.chunk.text) for item in answer.sources] == [(5, "beta")]


def test_ask_before_index_is_rejected():
    from beginner_pdf_rag.pipeline import PdfRag, RagError

    with pytest.raises(RagError, match="index"):
        PdfRag(FakeEmbedder(), FakeGenerator()).ask("question")


def test_ask_rejects_an_empty_query_embedding_batch(monkeypatch):
    from beginner_pdf_rag.pipeline import PdfRag, RagError

    class EmptyQueryEmbedder:
        def embed(self, texts):
            return [[1.0]] if texts[0] == "document" else []

    monkeypatch.setattr(
        "beginner_pdf_rag.pipeline.load_pdf", lambda _: [Page(1, "document")]
    )
    rag = PdfRag(EmptyQueryEmbedder(), FakeGenerator())
    rag.index("paper.pdf")

    with pytest.raises(RagError, match="query embedding"):
        rag.ask("question")


@pytest.mark.parametrize("question", ["", "   ", None, True])
def test_ask_rejects_blank_or_nonstring_questions_before_embedding(monkeypatch, question):
    from beginner_pdf_rag.pipeline import PdfRag, RagError

    monkeypatch.setattr(
        "beginner_pdf_rag.pipeline.load_pdf", lambda _: [Page(1, "alpha")]
    )
    rag = PdfRag(FakeEmbedder(), FakeGenerator())
    rag.index("paper.pdf")

    with pytest.raises(RagError, match="Question"):
        rag.ask(question)


@pytest.mark.parametrize("top_k", [True, False, 0, -1, 1.5, "1"])
def test_ask_rejects_impossible_top_k_values(monkeypatch, top_k):
    from beginner_pdf_rag.pipeline import PdfRag, RagError

    monkeypatch.setattr(
        "beginner_pdf_rag.pipeline.load_pdf", lambda _: [Page(1, "alpha")]
    )
    rag = PdfRag(FakeEmbedder(), FakeGenerator())
    rag.index("paper.pdf")

    with pytest.raises(RagError, match="top_k"):
        rag.ask("alpha", top_k=top_k)
