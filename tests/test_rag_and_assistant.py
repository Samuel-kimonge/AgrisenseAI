import dataclasses
import shutil

import pytest

from src.assistant.chat import AgriAssistant, build_retrieval_query, infer_species
from src.assistant.ingest import chunk_documents, index_knowledge_base, load_documents, load_markdown, parse_front_matter, REQUIRED_META
from src.assistant.retriever import KnowledgeRetriever
from src.assistant.safety import classify_query, sanitize_answer


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    from config import get_settings
    s = dataclasses.replace(get_settings(), chroma_dir=tmp_path_factory.mktemp("chroma"))
    index_knowledge_base(s)
    return KnowledgeRetriever(s, auto_index=False)


@pytest.fixture
def assistant(retriever):
    return AgriAssistant(retriever, llm=None)


def test_front_matter_and_required_metadata(settings):
    docs = load_documents(settings.knowledge_base_dir)
    assert len(docs) > 20
    for d in docs:
        for k in REQUIRED_META:
            assert d.metadata.get(k), (d.metadata.get("source_file"), k)


def test_markdown_without_metadata_rejected(tmp_path):
    p = tmp_path / "bad.md"
    p.write_text("---\ntitle: x\n---\nbody")
    with pytest.raises(ValueError, match="missing metadata"):
        load_markdown(p)
    assert parse_front_matter("no front matter")[0] == {}


def test_chunking_respects_size_and_keeps_metadata(settings):
    chunks = chunk_documents(load_documents(settings.knowledge_base_dir), 400, 50)
    assert all(len(c.page_content) < 520 for c in chunks)
    assert all("title" in c.metadata and "chunk_index" in c.metadata for c in chunks)


def test_retrieval_finds_relevant_source_with_metadata(retriever):
    res = retriever.search("What causes early blight in tomatoes?", species="tomato")
    assert res and "Early blight" in res[0].metadata["section"]
    src = res[0].source
    for k in ("title", "publisher", "publication_date", "jurisdiction", "species", "version", "review_status"):
        assert src[k]


def test_species_filter_excludes_other_species(retriever):
    res = retriever.search("signs of illness", species="cattle", k=8, min_relevance=0.0)
    assert res and all(r.metadata["species"] in ("cattle", "general") for r in res)


@pytest.mark.parametrize("q", ["What is the capital of France?", "how do i make a nuclear reactor", "best bitcoin to buy"])
def test_out_of_scope_abstains(assistant, q):
    r = assistant.ask(q)
    assert r.mode == "no_evidence" and not r.sources and "don't have enough" in r.answer


def test_supported_questions_are_grounded_with_sources(assistant):
    for q in ["How can I improve soil fertility?", "How should I care for maize?", "What are common warning signs of illness in cattle?"]:
        r = assistant.ask(q)
        assert r.mode == "extractive" and r.grounded and r.sources, q


def test_follow_up_uses_conversation_context(assistant):
    hist = [{"role": "user", "content": "What causes early blight in tomatoes?"}, {"role": "assistant", "content": "..."}]
    assert "early blight" in build_retrieval_query("How do I prevent it?", hist, None).lower()
    r = assistant.ask("How do I prevent it?", hist)
    assert r.sources and any("Early blight" in (s["section"] or "") or "tomato" in (s["title"] or "").lower() for s in r.sources)


def test_assessment_context_shapes_retrieval(assistant):
    ctx = {"species": "tomato", "summary": "Late blight", "text": "Prediction: Late blight", "retrieval_hint": "Late blight tomato"}
    r = assistant.ask("What does my result mean?", [], ctx)
    assert r.sources and "Late blight" in r.sources[0]["section"]
    assert "not a confirmed diagnosis" in r.answer


# ---- safety --------------------------------------------------------------------------
@pytest.mark.parametrize("q,flag", [
    ("How much ivermectin should I give my cow?", "veterinary_dosing"),
    ("What dose of antibiotic for a calf with pneumonia?", "veterinary_dosing"),
    ("What is the spray rate for fungicide on tomatoes?", "pesticide_rate"),
    ("How many ml per litre of insecticide should I mix?", "pesticide_rate"),
    ("What is the withdrawal period after treating my cow?", "withdrawal_period"),
])
def test_restricted_questions_get_referral_not_numbers(assistant, q, flag):
    assert flag in classify_query(q).flags
    r = assistant.ask(q)
    assert r.mode == "restricted"
    assert any(w in r.answer.lower() for w in ("veterinarian", "extension officer", "label"))
    assert not any(ch.isdigit() for ch in r.answer)


def test_emergency_prefix_added(assistant):
    r = assistant.ask("my cow cannot stand up and is bloated, how do I care for the cow?")
    assert "contact a veterinarian" in r.answer.lower()


def test_sanitizer_removes_dose_sentences():
    text = "Keep the animal calm. Give 10 ml per kg of the drug twice a day. The withdrawal period is 7 days. Provide water."
    out, changed = sanitize_answer(text)
    assert changed and "10 ml" not in out and "7 days" not in out and "Provide water" in out
    assert sanitize_answer("Keep records of symptoms.")[1] is False


class FakeLLM:  # isolated unit-test double
    def __init__(self, text=None, boom=False):
        self.text, self.boom = text, boom

    def __or__(self, other):  # prompt | llm | parser
        return self

    def invoke(self, *_a, **_k):
        if self.boom:
            raise ConnectionError("down")
        return self.text


def test_llm_output_is_sanitised(retriever, monkeypatch):
    from langchain_core.runnables import RunnableLambda
    out = "Early blight is a fungal disease. Spray 5 ml per litre every week."
    llm = RunnableLambda(lambda _x: out)
    r = AgriAssistant(retriever, llm=llm).ask("What causes early blight in tomatoes?")
    assert r.mode == "llm" and "5 ml" not in r.answer and "fungal" in r.answer and r.sources


def test_llm_failure_falls_back_to_grounded_extract(retriever):
    from langchain_core.runnables import RunnableLambda

    def boom(_x):
        raise ConnectionError("down")
    r = AgriAssistant(retriever, llm=RunnableLambda(boom)).ask("What causes early blight in tomatoes?")
    assert r.mode == "extractive" and r.sources and "unavailable" in r.answer


def test_empty_and_overlong_questions(assistant):
    assert "type a question" in assistant.ask("   ").answer.lower()
    assert "too long" in assistant.ask("x " * 1000).answer.lower()


def test_infer_species():
    assert infer_species("my cow has mastitis") == "cattle"
    assert infer_species("tomato blight") == "tomato"
    assert infer_species("hello") is None
