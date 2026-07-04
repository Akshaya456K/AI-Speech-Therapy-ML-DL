# Personalized Language Learning for Children with Speech Impediments
### Problem Statement 089 — full pipeline: ML → DL → NLP → SLM → LLD → GenAI → Agentic AI

## What's in here

```
project/
├── data/
│   └── speech_therapy_dataset.csv       # your original dataset
├── models/                              # your original ML/DL artifacts (untouched)
│   ├── speech_therapy_model.pkl
│   ├── speech_therapy_ann_model.keras
│   ├── scaler.pkl
│   └── target_encoder.pkl
├── src/
│   ├── data_utils.py                    # shared data loading/encoding
│   ├── ml_predictor.py                  # LEVEL: ML  (Random Forest + rule-based baseline)
│   ├── dl_predictor.py                  # LEVEL: DL  (Keras ANN)
│   ├── nlp_module.py                    # LEVEL: NLP (sentiment, NER, keywords, summarization)
│   ├── slm_module.py                    # LEVEL: SLM (personalized feedback generator)
│   ├── genai_module.py                  # LEVEL: GEN AI (stories, games, quizzes)
│   └── agent.py                         # LEVEL: AGENTIC AI (perceive→reason→plan→act→remember)
├── api/
│   └── app.py                           # Flask API exposing every layer above
├── docs/
│   ├── ML_DL_REVIEW.md                  # review of your original notebooks (read this first)
│   ├── LLD.md                           # architecture, ER diagram, API contract, scalability notes
│   └── openapi.yaml                     # machine-readable API spec
├── demo_pipeline.py                     # run this for a full end-to-end walkthrough
└── requirements.txt
```

Your original two notebooks (`Speech_Therapy_ML.ipynb`, `Speed_Therapy_DL.ipynb`)
are unmodified — nothing here rewrites your work, it reviews it
(`docs/ML_DL_REVIEW.md`) and builds forward from it.

## Quick start

```bash
cd project
pip install -r requirements.txt
python demo_pipeline.py          # runs ML -> NLP -> SLM -> GenAI -> Agentic AI on a sample child
python api/app.py                # start the REST API on :5000
```

## Read this first: `docs/ML_DL_REVIEW.md`
Both notebooks are technically correct (clean preprocessing, correct
train/test/scale order, correct metrics), **but** the dataset itself is
synthetically generated with deterministic rules
(`Speech_Error → Recommended_Exercise` and `Pronunciation_Score →
Improvement` are both 1:1 lookups, true for all 1000 rows) — so any
model, including a one-line rule-based lookup, hits ~100% accuracy. This
is a dataset property, not a bug in your code, but it's important to
name explicitly in your report since "data leakage" is literally in the
problem statement's own "Common Mistakes" list. The review doc also
flags that the uploaded `.pkl`/`.csv` don't reproduce against each other
(likely different data pulls) — worth re-saving your artifacts from a
run against the final CSV before submission.

## How each level builds on the last
1. **ML** (`ml_predictor.py`) — Random Forest recommends an exercise
   from the child's profile; a one-line rule-based baseline is included
   for direct comparison (see review doc).
2. **DL** (`dl_predictor.py`) — a Dense ANN (64→32→16→softmax),
   reproducing your notebook's architecture; MLP is the right choice
   here since the data has no sequence/time structure (no need for
   RNN/LSTM).
3. **NLP** (`nlp_module.py`) — turns unstructured therapist/parent notes
   into structured signal: sentiment, phoneme/score/session entities,
   TF-IDF keywords, TextRank summary.
4. **SLM** (`slm_module.py`) — generates short, personalized
   feedback/explanations via a retrieval + template engine (a legitimate
   SLM-substitute pattern given this sandbox has no internet access to
   download a real fine-tuned model). `call_anthropic_slm()` shows the
   one-function swap to a real hosted model in production.
5. **LLD** (`docs/LLD.md`) — architecture diagram, ER diagram, API
   contract (`docs/openapi.yaml`), scalability/security notes.
6. **GenAI** (`genai_module.py`) — generates phoneme-rich stories,
   articulation games, and adaptive quizzes, personalized by interest
   and age, with a lightweight content-quality check.
7. **Agentic AI** (`agent.py`) — the full autonomous loop: perceives the
   child's profile + latest note, reasons using ML + NLP together, plans
   a session using SLM + GenAI, acts by producing and logging a session
   plan, and remembers history per child. Includes an explicit
   human-in-the-loop safety gate (`needs_human_review`) that fires on
   low ML confidence, high severity, or negative note sentiment — the
   agent never silently auto-finalizes a plan in those cases.

## Design constraint you should know about
This sandbox has no internet access to model hubs (Hugging Face, OpenAI,
etc.), so the SLM and GenAI layers are built with fully-offline,
inspectable techniques (retrieval + template generation) rather than a
literal fine-tuned transformer. Every function in `nlp_module.py` and
`slm_module.py` is marked with a `# --- swap point ---` comment showing
exactly where to plug in a real model (spaCy/BERT for NLP, a hosted
LLM/SLM API for generation) without touching `agent.py` or `api/app.py`.
If you have access to the Anthropic API (or any LLM API) in your own
environment, `slm_module.call_anthropic_slm()` is a ready-to-use
integration point.
