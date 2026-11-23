import dataclasses

import pytest

from src.assistant.chat import AgriAssistant, build_retrieval_query
from src.assistant.ingest import chunk_documents, index_knowledge_base, load_documents, load_markdown, REQUIRED_META
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


def test_documents_have_required_metadata():
    from config import get_settings
    docs = load_documents(get_settings().knowledge_base_dir)
    assert len(docs) > 20
    for d in docs:
        for k in REQUIRED_META:
            assert d.metadata.get(k), (d.metadata.get("source_file"), k)


def test_missing_front_matter_rejected(tmp_path):
    p = tmp_path / "bad.md"
    p.write_text("no metadata here")
    with pytest.raises(ValueError):
        load_markdown(p)


def test_chunking_preserves_metadata_and_size():
    from config import get_settings
    chunks = chunk_documents(load_documents(get_settings().knowledge_base_dir), 700, 100)
    assert all(len(c.page_content) < 900 for c in chunks)
    assert all("chunk_index" in c.metadata and c.metadata["title"] for c in chunks)


@pytest.mark.parametrize("q,expected", [
    ("What causes early blight in tomatoes?", "Early blight"),
    ("How can I improve soil fertility?", "Improving Soil"),
    ("What are common warning signs of illness in cattle?", "Cattle"),
])
def test_retrieval_finds_relevant_source(retriever, q, expected):
    res = retriever.search(q)
    assert res and any(expected in (c.metadata["title"] + c.metadata["section"]) for c in res[:3])


def test_species_filter(retriever):
    res = retriever.search("feeding and water", species="cattle")
    assert res and all(c.metadata["species"] in ("cattle", "general") for c in res)


@pytest.mark.parametrize("q", ["What is the capital of France?", "how do i make a nuclear reactor", "best bitcoin to buy"])
def test_out_of_scope_questions_abstain(assistant, q):
    r = assistant.ask(q)
    assert r.mode == "no_evidence" and not r.sources and not r.grounded


def test_answer_has_sources_with_metadata(assistant):
    r = assistant.ask("How do I prevent tomato leaf diseases?")
    assert r.mode == "extractive" and r.grounded and r.sources
    s = r.sources[0]
    assert s["publisher"] and s["review_status"] and s["publication_date"]


@pytest.mark.parametrize("q,flag", [
    ("How much ivermectin should I give my cow?", "veterinary_dosing"),
    ("What is the spray rate of fungicide per litre for blight?", "pesticide_rate"),
    ("What is the withdrawal period for milk after antibiotics?", "withdrawal_period"),
])
def test_restricted_questions_refused(assistant, q, flag):
    assert flag in classify_query(q).flags
    r = assistant.ask(q)
    assert r.mode == "restricted"
    assert "veterinarian" in r.answer.lower() or "extension officer" in r.answer.lower()
    assert not any(ch.isdigit() for ch in r.answer)


def test_emergency_prefix_added(assistant):
    r = assistant.ask("my cow is bloated and can't breathe")
    assert "contact a veterinarian" in r.answer.lower()


def test_sanitize_removes_dose_sentences():
    out, changed = sanitize_answer("Keep the animal calm. Give 10 ml per kg of the drug. Call a vet.")
    assert changed and "10 ml" not in out and "Call a vet" in out
    assert sanitize_answer("Keep it cool.")[1] is False


def fake_llm(text=None, fail=False):
    """Isolated unit-test stand-in for a chat model (never used in the app)."""
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    def _run(_):
        if fail:
            raise RuntimeError("network down")
        return AIMessage(content=text)
    return RunnableLambda(_run)


def test_llm_output_is_sanitised(retriever):
    llm = fake_llm("Early blight makes brown spots. Spray 5 ml per litre every week.")
    r = AgriAssistant(retriever, llm).ask("What causes early blight in tomatoes?")
    assert r.mode == "llm" and "5 ml" not in r.answer and "Early blight" in r.answer and r.sources


def test_llm_failure_falls_back_to_grounded_extracts(retriever):
    r = AgriAssistant(retriever, fake_llm(fail=True)).ask("What causes early blight in tomatoes?")
    assert r.mode == "extractive" and r.grounded and r.sources


def test_followup_query_uses_history_and_context():
    hist = [{"role": "user", "content": "What causes early blight?"}, {"role": "assistant", "content": "..."}]
    q = build_retrieval_query("How do I prevent it?", hist, {"retrieval_hint": "Early blight tomato"})
    assert "early blight" in q.lower() and "prevent" in q.lower()


def test_followup_retrieves_topic(assistant):
    hist = [{"role": "user", "content": "What causes early blight in tomatoes?"}]
    r = assistant.ask("How do I prevent it?", hist)
    assert r.mode == "extractive" and any("blight" in (s["section"] or "").lower() or "tomato" in s["title"].lower() for s in r.sources)
