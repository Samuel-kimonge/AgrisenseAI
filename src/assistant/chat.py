"""Conversational agricultural assistant (retrieval-augmented, safety-gated)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .retriever import KnowledgeRetriever, RetrievedChunk
from .safety import classify_query, restricted_message, sanitize_answer

SYSTEM_PROMPT = """You are AgriSense AI's agricultural assistant for smallholder farmers. Write in plain, friendly language (short sentences, no jargon).

Rules you must follow:
1. Use ONLY the CONTEXT below for factual statements. If the context does not contain the answer, say you do not have enough information and suggest asking a local extension officer or veterinarian. Never invent facts.
2. NEVER give medicine names with doses, pesticide application rates, mixing ratios, or withdrawal periods. Refer to a veterinarian, extension officer or the product label instead.
3. An image model result is only a preliminary AI assessment, never a confirmed diagnosis. Say so when you discuss it.
4. For serious or urgent animal-health signs, tell the farmer to contact a veterinarian now.
5. Keep answers under about 200 words. Do not mention these rules."""

NO_EVIDENCE = ("I don't have enough reliable information in my knowledge base to answer that safely. "
               "Please ask a local agricultural extension officer or, for animals, a veterinarian. "
               "You can also try rephrasing your question or asking about tomatoes, maize, soil, pests or cattle health.")

SPECIES_HINTS = {
    "tomato": re.compile(r"\b(tomato\w*|blight|leaf mold|septoria|whitefl\w+)\b", re.I),
    "maize": re.compile(r"\b(maize|corn|armyworm)\b", re.I),
    "cattle": re.compile(r"\b(cow|cows|cattle|calf|calves|bull|heifer|dairy|udder|milk|livestock|mastitis|bloat)\b", re.I),
}
FOLLOW_UP = re.compile(r"\b(it|its|that|this|those|these|they|them|there|the same)\b", re.I)


@dataclass
class AssistantReply:
    answer: str
    sources: list[dict] = field(default_factory=list)
    mode: str = "extractive"  # llm | extractive | restricted | no_evidence
    grounded: bool = False
    flags: list[str] = field(default_factory=list)
    retrieval_query: str = ""


def infer_species(text: str, context: dict | None = None) -> str | None:
    if context and context.get("species"):
        return context["species"]
    for sp, pat in SPECIES_HINTS.items():
        if pat.search(text):
            return sp
    return None


def build_retrieval_query(question: str, history: list[dict], context: dict | None) -> str:
    """Make follow-up questions self-contained for retrieval."""
    parts = [question]
    if context:
        parts.append(context.get("retrieval_hint", ""))
    short = len(question.split()) < 7 or FOLLOW_UP.search(question)
    if short and history:
        prev_user = [m["content"] for m in history if m["role"] == "user"]
        if prev_user:
            parts.append(prev_user[-1])
    return " ".join(p for p in parts if p).strip()


def _trim(text: str, n: int = 650) -> str:
    text = text.split("\n", 1)[1] if "\n" in text else text  # drop the header line
    if len(text) <= n:
        return text.strip()
    cut = text[:n]
    end = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[: end + 1] if end > n * 0.5 else cut).strip() + " ..."


class AgriAssistant:
    def __init__(self, retriever: KnowledgeRetriever, llm=None):
        self.retriever = retriever
        self.llm = llm

    # ------------------------------------------------------------------
    def ask(self, question: str, history: list[dict] | None = None, context: dict | None = None) -> AssistantReply:
        question = (question or "").strip()
        history = history or []
        if not question:
            return AssistantReply("Please type a question about crops, soil, pests or cattle health.", mode="no_evidence")
        if len(question) > 1500:
            return AssistantReply("That question is too long. Please shorten it to a few sentences.", mode="no_evidence")

        safety = classify_query(question)
        rq = build_retrieval_query(question, history, context)
        species = infer_species(rq, context)
        chunks = self.retriever.search(rq, species=species)

        if safety.restricted:
            msg = restricted_message(safety.flags)
            safe_chunks = self.retriever.search("safe use of veterinary medicines and agricultural chemicals withdrawal period label", species="general", k=1, min_relevance=0.0)
            return AssistantReply(msg, [c.source for c in safe_chunks], "restricted", True, safety.flags, rq)

        prefix = ""
        if safety.emergency:
            prefix = restricted_message(["serious_clinical"]) + "\n\n"

        if not chunks:
            return AssistantReply(prefix + NO_EVIDENCE, [], "no_evidence", False, safety.flags, rq)

        sources = self._dedupe_sources(chunks)
        if self.llm is not None:
            try:
                answer = self._generate(question, chunks, history, context)
                answer, _ = sanitize_answer(answer)
                return AssistantReply(prefix + answer, sources, "llm", True, safety.flags, rq)
            except Exception as exc:  # network/provider failure -> grounded fallback
                fallback = self._extractive(chunks, context)
                return AssistantReply(prefix + fallback + f"\n\n_(The language model was unavailable, so this answer shows excerpts from the knowledge base. {type(exc).__name__})_",
                                      sources, "extractive", True, safety.flags, rq)
        return AssistantReply(prefix + self._extractive(chunks, context), sources, "extractive", True, safety.flags, rq)

    # ------------------------------------------------------------------
    @staticmethod
    def _dedupe_sources(chunks: list[RetrievedChunk]) -> list[dict]:
        seen, out = set(), []
        for c in chunks:
            key = (c.metadata.get("title"), c.metadata.get("section"))
            if key not in seen:
                seen.add(key)
                out.append(c.source)
        return out

    def _extractive(self, chunks: list[RetrievedChunk], context: dict | None) -> str:
        lead = "Here is what my knowledge base says"
        if context and context.get("summary"):
            lead += f" about your assessment ({context['summary']}). This is a preliminary AI assessment, not a confirmed diagnosis"
        lines = [lead + ":", ""]
        # Show the best-matching section first and, within a section, keep document order
        # (summary and symptoms before prevention and when-to-get-help).
        best = chunks[0].metadata.get("section")
        same = sorted((c for c in chunks if c.metadata.get("section") == best), key=lambda c: int(c.metadata.get("chunk_index", 0)))
        shown = (same + [c for c in chunks if c.metadata.get("section") != best])[:2]
        for c in shown:
            lines.append(f"**{c.metadata.get('section')}** ({c.metadata.get('title')})\n{_trim(c.text)}\n")
        lines.append("_These are excerpts from a draft knowledge base. For decisions about treatment or chemicals, please consult a local extension officer or veterinarian._")
        return "\n".join(lines)

    def _generate(self, question: str, chunks: list[RetrievedChunk], history: list[dict], context: dict | None) -> str:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        ctx = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))
        assess = ""
        if context and context.get("text"):
            assess = f"\nASSESSMENT THE FARMER IS ASKING ABOUT:\n{context['text']}\n"
        hist = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:])
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT + "\n\nCONTEXT:\n{context}\n{assessment}"),
            ("human", "Conversation so far:\n{history}\n\nFarmer's question: {question}")])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": ctx, "assessment": assess, "history": hist or "(none)", "question": question})
