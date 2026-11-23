"""Knowledge-base ingestion: load -> extract text/metadata -> chunk -> embed -> index (ChromaDB)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embeddings import get_embeddings

COLLECTION = "agrisense_kb"
FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
REQUIRED_META = ("title", "publisher", "publication_date", "jurisdiction", "species", "version", "review_status")


def parse_front_matter(text: str) -> tuple[dict, str]:
    m = FRONT_MATTER.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, text[m.end():]


def _split_sections(body: str) -> list[tuple[str, str]]:
    parts = re.split(r"^##\s+(.*)$", body, flags=re.M)
    sections = []
    if parts[0].strip():
        sections.append(("Overview", parts[0].strip()))
    for i in range(1, len(parts), 2):
        sections.append((parts[i].strip(), parts[i + 1].strip()))
    return sections


def load_markdown(path: Path) -> list[Document]:
    meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise ValueError(f"{path.name}: missing metadata fields {missing}")
    docs = []
    for section, text in _split_sections(body):
        docs.append(Document(page_content=text, metadata={**meta, "section": section, "source_file": path.name}))
    return docs


def load_condition_json(path: Path) -> list[Document]:
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data["_meta"]
    docs = []
    for key, c in data["conditions"].items():
        def bullets(label, items):
            return f"{label}: " + "; ".join(items) + "." if items else ""
        text = "\n".join(x for x in [
            c["summary"],
            f"Type: {c['category']}." + (f" Cause: {c['cause_organism']}." if c.get("cause_organism") else ""),
            bullets("Symptoms", c["symptoms"]),
            bullets("Possible causes", c["possible_causes"]),
            bullets("Prevention", c["prevention"]),
            bullets("Management", c["management"]),
            bullets("Get expert help if", c["get_help_if"]),
        ] if x)
        docs.append(Document(page_content=text, metadata={
            **{k: v for k, v in meta.items()}, "section": c["name"], "condition_key": key, "source_file": path.name,
            "title": f"{meta['title']} - {c['name']}"}))
    return docs


def load_documents(kb_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for p in sorted(Path(kb_dir).glob("*.md")):
        docs += load_markdown(p)
    cj = Path(kb_dir) / "tomato_diseases.json"
    if cj.exists():
        docs += load_condition_json(cj)
    if not docs:
        raise FileNotFoundError(f"No knowledge-base documents found in {kb_dir}")
    return docs


def chunk_documents(docs: list[Document], chunk_size: int = 700, overlap: int = 100) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap,
                                              separators=["\n\n", "\n", ". ", " "])
    out: list[Document] = []
    for d in docs:
        pieces = splitter.split_text(d.page_content)
        for i, piece in enumerate(pieces):
            header = f"{d.metadata['title']} | {d.metadata['section']}"
            md = {k: str(v) for k, v in d.metadata.items()}
            md.update({"chunk_index": i, "n_chunks": len(pieces)})
            out.append(Document(page_content=f"{header}\n{piece}", metadata=md))
    return out


def kb_fingerprint(kb_dir: Path, chunk_size: int, overlap: int, provider: str) -> str:
    h = hashlib.sha256(f"{chunk_size}|{overlap}|{provider}".encode())
    for p in sorted(Path(kb_dir).glob("*")):
        if p.suffix in {".md", ".json"} and p.name != "livestock_rules.json":
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def get_store(settings, embeddings=None):
    from langchain_chroma import Chroma
    return Chroma(collection_name=COLLECTION, embedding_function=embeddings or get_embeddings(settings),
                  persist_directory=str(settings.chroma_dir), collection_metadata={"hnsw:space": "cosine"})


def index_knowledge_base(settings, embeddings=None, reset: bool = True) -> dict:
    """(Re)build the Chroma index from the knowledge-base folder."""
    embeddings = embeddings or get_embeddings(settings)
    chunks = chunk_documents(load_documents(settings.knowledge_base_dir), settings.chunk_size, settings.chunk_overlap)
    store = get_store(settings, embeddings)
    if reset:
        try:
            store.delete_collection()
        except Exception:
            pass
        store = get_store(settings, embeddings)
    ids = [f"{c.metadata['source_file']}::{c.metadata['section']}::{c.metadata['chunk_index']}" for c in chunks]
    store.add_documents(chunks, ids=ids)
    fp = kb_fingerprint(settings.knowledge_base_dir, settings.chunk_size, settings.chunk_overlap, settings.embedding_provider)
    (Path(settings.chroma_dir) / "kb_fingerprint.txt").write_text(fp)
    return {"documents": len({c.metadata["source_file"] for c in chunks}), "chunks": len(chunks), "fingerprint": fp}


def ensure_index(settings, embeddings=None) -> dict | None:
    """Index only if missing or stale. Returns stats when it (re)indexed, else None."""
    fp_file = Path(settings.chroma_dir) / "kb_fingerprint.txt"
    fp = kb_fingerprint(settings.knowledge_base_dir, settings.chunk_size, settings.chunk_overlap, settings.embedding_provider)
    if fp_file.exists() and fp_file.read_text() == fp:
        return None
    return index_knowledge_base(settings, embeddings)
