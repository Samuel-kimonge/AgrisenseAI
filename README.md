# AgriSense AI

AI-assisted agricultural health prototype (Streamlit + PyTorch):

1. **Plant disease detection** - upload/take a photo of a tomato leaf; a trained MobileNetV3 classifier returns a *preliminary* assessment with calibrated confidence, alternatives, and an abstention mechanism for unclear or unsupported images.
2. **Livestock health assessment** - structured cattle symptom questionnaire -> rule-based triage (urgency, possible concerns, next steps, vet referral). No animal image diagnosis, no medicine or dose recommendations.
3. **AI agricultural assistant** - RAG chat (LangChain + ChromaDB) over a curated knowledge base, with source attribution, follow-up questions, and strict safety restrictions.
4. **Assessment history** - SQLite; open past assessments and ask follow-up questions about each.

> **Status: prototype.** Results are preliminary AI output, not diagnoses. See [Known limitations](#known-limitations).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # optional: configure an LLM provider
streamlit run app.py
```
Python 3.11+ (developed and tested on 3.12). CPU-only is fine (8 GB RAM recommended).
If no trained model is present the app still starts, clearly says the model is unavailable and **never returns made-up predictions**.

## Train the plant model

PlantVillage tomato images are **not** shipped in this repo. Get the `raw/color` tomato folders (10 classes) from
<https://github.com/spMohanty/PlantVillage-Dataset> (sparse clone below) and place them in `data/raw/plantvillage_tomato/`:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset.git pv
cd pv && git sparse-checkout set --no-cone '/raw/color/Tomato*' && cd ..
mkdir -p data/raw && cp -r pv/raw/color data/raw/plantvillage_tomato
```
Pretrained backbone weights (timm MobileNetV3-Large, ImageNet) go in `models/pretrained/`:
```bash
curl -L -o models/pretrained/mobilenetv3_large_100_ra-f55367f5.pth \
  https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth
```
(With normal internet access you may instead let timm download weights; see `src/ml/model.py`.)

```bash
python scripts/prepare_dataset.py                 # discovery, integrity checks, dedupe, stratified split
python scripts/train.py --head-epochs 2 --finetune-epochs 4 --train-cap 350   # CPU-friendly
#   --resume models/checkpoints/best.pt           # continue an interrupted run
python scripts/make_model_card.py                 # writes reports/model_cards/tomato_classifier.md from real metrics
```
Outputs: `models/production/{model.pt,metadata.json}` (weights, class order, temperature, abstention thresholds, metrics) and `reports/evaluation/*` (test/val reports, confusion matrix, classification report, OOD check, training history). The test split is used only for the final report.

## Assistant (RAG) configuration

Everything is configured through environment variables (`.env.example`); no keys are in the code.

| Setting | Options | Notes |
|---|---|---|
| `LLM_PROVIDER` | `none` (default), `anthropic`, `openai`, `ollama` | `none` -> grounded **extractive** answers from the knowledge base. Install the matching package: `langchain-anthropic`, `langchain-openai` or `langchain-ollama`. |
| `EMBEDDING_PROVIDER` | `hashing` (default, offline), `sentence-transformers` | Hashing is lexical; neural embeddings handle paraphrases better. |

The knowledge base lives in `data/knowledge_base/` (Markdown with metadata front matter + `tomato_diseases.json`). The index in `chroma_db/` is rebuilt automatically when files change, or run `python scripts/index_knowledge_base.py`. **All documents are drafts pending review** by agronomists/veterinarians - see their `review_status` metadata.

## Tests

```bash
pytest -q
```
Covers image validation/preprocessing, dataset checks, model loading/prediction/abstention (unit tests use a tiny random model; integration tests use the real trained model on held-out images and are skipped if none is installed), livestock rules, RAG retrieval and safety, SQLite, history, recommendations, and Streamlit page smoke tests.

## Project layout

```
app.py, config.py            entry point (navigation + home), settings from env
pages/                       Streamlit pages (thin presentation)
src/ui/                      shared components/theme            [presentation layer]
src/services/                assessment, recommendation, assistant glue   [application layer]
src/ml/                      dataset, model, train, evaluate, inference   [ML layer]
src/utils/                   image quality checks, safe image storage     [image-processing layer]
src/livestock/               schema + rule-based triage                   [livestock layer]
src/assistant/               embeddings, ingest, retriever, safety, LLM, chat   [assistant/RAG + safety]
src/database/                SQLite migrations + parameterised repository [database layer]
data/knowledge_base/         curated docs + livestock rules (JSON)
reports/, models/, tests/, scripts/
```

## Safety design

- Abstention: quality gate (blur/dark/tiny/not-leaf) -> calibrated confidence + margin + max-logit thresholds -> "uncertain", never a forced label.
- Temperature-scaled probabilities are reported as *model confidence*, not diagnostic probability.
- The assistant refuses medicine doses, pesticide rates and withdrawal periods (query filter **and** output sanitiser), adds emergency referral for serious signs, and abstains when retrieval evidence is weak.
- Livestock rules are data (`livestock_rules.json`), conservative, only ever escalate urgency, and are flagged DRAFT until a veterinarian reviews them.
- Uploads are validated (type, size, decode), re-encoded as JPEG, stored on disk with random names; the DB holds only relative paths; all SQL is parameterised.

## Known limitations

See also `reports/model_cards/tomato_classifier.md` and `DATASET.md`.

- Trained on PlantVillage (controlled photos). **Field accuracy is unmeasured and expected to be lower.** Collect and label local field photos before any real deployment.
- Random splits may leak near-duplicate leaves between train and test, inflating test scores.
- The model cannot reliably recognise "not tomato" leaves by itself; abstention helps but is imperfect (measured in the model card).
- Livestock triage rules and all knowledge-base content are drafts that have **not** been expert-reviewed. No animal image model exists in this version.
- Hashing embeddings can miss paraphrased questions; use neural embeddings and/or an LLM for better language coverage.
- Single-user, local SQLite; no authentication.
- Not built (by design): drones, satellite, IoT, irrigation control, ERP, marketplace, finance, outbreak/yield forecasting, other crops/species. Extension points: `src/ml/classes.py` + `scripts/train.py` for crops, `livestock_rules.json` for species/rules, new files in `data/knowledge_base/`.
# AgrisenseAI
