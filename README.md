# Graphically Wrong

> Partial credit grading powered by knowledge graphs. Not just *is it right* — but *how wrong is it?*

Graphically Wrong reads your textbook, builds a concept graph from it, then scores student answers by measuring how far each answer sits from the correct one in that graph. A student who writes the contrasting term gets more credit than one who writes something from a completely different domain.

---

## How it works

1. **PDF → Graph** — The textbook is converted to a knowledge graph: concepts as nodes, typed relationships (is-a, part-of, influences, etc.) as weighted edges
2. **Answer → Concepts** — An LLM maps each student answer to nodes in the graph
3. **Distance → Score** — Weighted shortest-path distance between the student's concepts and the correct answer's concepts is normalized into a score between 0 and 1

---

## Getting started

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set your API key

```bash
export ELPAI_API_KEY="your_key_here"
```

### Run the app

```bash
streamlit run app.py
```

---

## Usage

Upload three files in the web UI:

| File | Format | Required columns |
|------|--------|-----------------|
| Textbook | `.pdf` | — |
| Questions & correct answers | `.csv` | `question_id`, `question`, `correct_answer` |
| Student responses | `.csv` | `student_id`, `question_id`, `answer` |

The knowledge graph is built once per textbook and cached locally — subsequent runs on the same PDF are fast.

Download CSV templates from the sidebar if you need a starting point.

---

## Scoring

| Score | Meaning |
|-------|---------|
| `1.0` | Conceptually identical to the correct answer |
| `~0.85` | Same conceptual family (e.g. contrasting term) |
| `~0.70` | Related concept, different schema |
| `~0.50` | Different domain |
| `0.0` | No mapped concepts or completely disconnected |

**Max Distance** and **Penalty Power** can be tuned in the sidebar to make scoring more lenient or strict.

---

## Stack

- [Streamlit](https://streamlit.io) — UI
- [NetworkX](https://networkx.org) — graph construction and path computation
- [OpenAI-compatible API](https://chat.elpai.org) — PDF parsing, graph extraction, concept mapping
- [PyMuPDF](https://pymupdf.readthedocs.io) — PDF to Markdown conversion
