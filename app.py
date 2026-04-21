"""
Knowledge Graph Answer Scorer — Streamlit Web App
Run: streamlit run app.py
"""

import json
import hashlib
import sys
import tempfile
import io
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KG Answer Scorer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #f4f6f9; }
.block-container { padding-top: 1.5rem; max-width: 1100px; }
div[data-testid="stFileUploader"] { border: 2px dashed #ced4da; border-radius: 10px; padding: 0.5rem 1rem; background: white; }
.result-card { background: white; border-radius: 10px; padding: 1.5rem; margin-bottom: 1rem; border: 1px solid #dee2e6; }
thead tr th { background-color: #f1f3f5 !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# KG cache
# ──────────────────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / ".kg_cache"
CACHE_DIR.mkdir(exist_ok=True)

def _pdf_hash(pdf_bytes: bytes) -> str:
    return hashlib.md5(pdf_bytes).hexdigest()[:12]

def _kg_cache_path(h: str) -> Path:
    return CACHE_DIR / f"{h}_kg.json"

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar: scoring settings + score preview + templates
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("Scoring")
    max_dist = st.slider(
        "Max Distance", 5.0, 20.0, 10.0, 0.5,
        help="Reference distance for normalization. Lower = harsher overall.",
    )
    power = st.slider(
        "Penalty Power", 1.0, 3.0, 1.3, 0.1,
        help="1.0 = linear penalty. Higher values penalize far answers more steeply.",
    )

    st.divider()
    st.subheader("Score Preview")
    st.caption("Expected scores at current settings:")
    for d, label in [(0, "Correct answer"), (2, "Same schema"), (3.5, "Related"), (5.5, "Different domain")]:
        sim = max(0.0, 1.0 - (min(d, max_dist) / max_dist) ** power)
        filled = int(sim * 10)
        bar = "█" * filled + "░" * (10 - filled)
        st.caption(f"`{bar}` **{sim:.2f}**  {label}")

    st.divider()
    st.subheader("CSV Templates")
    st.caption("Download and fill in these templates:")

    qa_template = "question_id,question,correct_answer\nQ01,\"What type of knowledge refers to facts that can be explicitly stated?\",\"declarative knowledge\"\nQ02,\"What cognitive system temporarily holds and manipulates information?\",\"working memory\"\n"
    st.download_button("📥 Questions template", qa_template, "questions.csv", "text/csv", width='stretch')

    resp_template = "student_id,question_id,answer\nS1,Q01,\"procedural knowledge\"\nS1,Q02,\"short term memory\"\nS2,Q01,\"mental picture\"\nS2,Q02,\"long term memory\"\n"
    st.download_button("📥 Responses template", resp_template, "responses.csv", "text/csv", width='stretch')

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("Knowledge Graph Answer Scorer")
st.markdown(
    "Scores student answers by measuring semantic distance in a knowledge graph "
    "built from your textbook. Wrong answers that are *conceptually closer* to the "
    "correct answer receive higher partial credit."
)
st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# File upload — three columns
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Upload Files")
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown(
        '<span style="font-weight:600;font-size:0.95rem;">① Textbook PDF</span><br>'
        '<span style="font-size:0.75rem;color:#6c757d;font-family:monospace;">&nbsp;</span>',
        unsafe_allow_html=True,
    )
    pdf_file = st.file_uploader(
        "Textbook PDF", type=["pdf"], label_visibility="collapsed", key="pdf_upload"
    )
    if pdf_file:
        pdf_bytes = pdf_file.read(); pdf_file.seek(0)
        h = _pdf_hash(pdf_bytes)
        kg_path = _kg_cache_path(h)
        if kg_path.exists():
            kg_info = json.loads(kg_path.read_text(encoding="utf-8"))
            st.success(
                f"✓ KG already built\n\n"
                f"{len(kg_info['nodes'])} nodes · {len(kg_info['edges'])} edges"
            )
        else:
            st.info("KG will be built on first run (~3–5 min, saved for reuse)")

with col2:
    st.markdown(
        '<span style="font-weight:600;font-size:0.95rem;">② Questions &amp; Correct Answers</span><br>'
        '<span style="font-size:0.75rem;color:#6c757d;font-family:monospace;">question_id, question, correct_answer</span>',
        unsafe_allow_html=True,
    )
    qa_file = st.file_uploader(
        "Questions CSV", type=["csv"], label_visibility="collapsed", key="qa_upload"
    )
    if qa_file:
        try:
            qa_df = pd.read_csv(qa_file); qa_file.seek(0)
            required = {"question_id", "question", "correct_answer"}
            missing = required - set(qa_df.columns)
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                st.success(f"✓ {len(qa_df)} questions")
                st.dataframe(
                    qa_df[["question_id", "correct_answer"]],
                    width='stretch', hide_index=True, height=130,
                )
        except Exception as e:
            st.error(str(e))

with col3:
    st.markdown(
        '<span style="font-weight:600;font-size:0.95rem;">③ Student Answers</span><br>'
        '<span style="font-size:0.75rem;color:#6c757d;font-family:monospace;">student_id, question_id, answer</span>',
        unsafe_allow_html=True,
    )
    answers_files = st.file_uploader(
        "Responses CSV", type=["csv"], label_visibility="collapsed",
        key="answers_upload", accept_multiple_files=True,
    )
    if answers_files:
        try:
            frames = []
            for f in answers_files:
                frames.append(pd.read_csv(f)); f.seek(0)
            answers_df = pd.concat(frames, ignore_index=True)
            required = {"student_id", "question_id", "answer"}
            missing = required - set(answers_df.columns)
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                n_students = answers_df["student_id"].nunique()
                st.success(f"✓ {n_students} students · {len(answers_df)} answers ({len(answers_files)} file{'s' if len(answers_files) > 1 else ''})")
                st.dataframe(
                    answers_df.head(4),
                    width='stretch', hide_index=True, height=130,
                )
        except Exception as e:
            st.error(str(e))

# ──────────────────────────────────────────────────────────────────────────────
# Score button
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
all_uploaded = pdf_file and qa_file and answers_files
run = st.button(
    "▶  Score Answers",
    type="primary",
    disabled=not all_uploaded,
    width='stretch',
)

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────
if run:
    from pdf_to_md import pdf_to_markdown
    from md_to_kg import chunk_markdown, extract_graph_from_chunk, build_graph, export_json
    from answer_scorer import load_graph, extract_concepts, pairwise_distances, compute_similarity
    from llm_client import get_client, DEFAULT_MODEL

    # Re-read files (seek after earlier reads)
    pdf_file.seek(0); pdf_bytes = pdf_file.read()
    qa_file.seek(0);  qa_df     = pd.read_csv(qa_file)
    frames = []
    for f in answers_files:
        f.seek(0); frames.append(pd.read_csv(f))
    answers_df = pd.concat(frames, ignore_index=True)

    h       = _pdf_hash(pdf_bytes)
    kg_path = _kg_cache_path(h)
    client  = get_client()

    # ── Build KG if not cached ──────────────────────────────────────────────
    GRAPH_MSGS = [
        "Connecting the dots…",
        "Finding relationships…",
        "Mapping concepts…",
        "Weaving the graph…",
        "Linking ideas…",
        "Almost there…",
    ]

    if not kg_path.exists():
        with st.status("Building knowledge graph from textbook…", expanded=True) as status:
            st.write("📄 **Step 1/2** — Reading your textbook…")
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(pdf_bytes)
                tmp_pdf = Path(f.name)
            tmp_md = tmp_pdf.with_suffix(".md")
            with st.spinner("Converting PDF to text… 📖"):
                pdf_to_markdown(str(tmp_pdf), str(tmp_md))
            md_text = tmp_md.read_text(encoding="utf-8")
            tmp_pdf.unlink(); tmp_md.unlink()
            st.write("✅ **Step 1/2** — Textbook converted!")

            chunks = chunk_markdown(md_text)
            st.write(f"🔗 **Step 2/2** — Building concept graph ({len(chunks)} chunks)…")
            prog = st.progress(0, text="Starting…")
            all_data = []
            for i, chunk in enumerate(chunks):
                result = extract_graph_from_chunk(client, chunk, DEFAULT_MODEL)
                all_data.append(result)
                msg = GRAPH_MSGS[i % len(GRAPH_MSGS)]
                prog.progress((i + 1) / len(chunks), text=f"{msg} ({i+1}/{len(chunks)})")

            G_build = build_graph(all_data)
            export_json(G_build, kg_path)
            status.update(
                label=f"✓ Knowledge graph built — {G_build.number_of_nodes()} nodes, {G_build.number_of_edges()} edges",
                state="complete",
            )
            st.balloons()

    # ── Score answers ───────────────────────────────────────────────────────
    G = load_graph(kg_path)
    vocab = [{"id": n, "label": d.get("label", n)} for n, d in G.nodes(data=True)]

    merged = answers_df.merge(
        qa_df[["question_id", "question", "correct_answer"]],
        on="question_id", how="left",
    )

    total = len(merged)
    prog  = st.progress(0, text="Scoring answers…")
    rows  = []

    for i, row in enumerate(merged.itertuples(index=False)):
        correct_concepts = extract_concepts(client, DEFAULT_MODEL, row.question, row.correct_answer, vocab)
        student_concepts = extract_concepts(client, DEFAULT_MODEL, row.question, row.answer, vocab)

        dists    = pairwise_distances(G, correct_concepts, student_concepts)
        avg_dist = round(sum(dists) / len(dists), 3) if dists else None
        kg_score = compute_similarity(G, correct_concepts, student_concepts, max_dist, power)

        binary = int(str(row.answer).strip().lower() == str(row.correct_answer).strip().lower())

        rows.append({
            "student_id":        row.student_id,
            "question_id":       row.question_id,
            "correct_answer":    row.correct_answer,
            "student_answer":    row.answer,
            "binary":            binary,
            "kg_score":          kg_score,
            "avg_dist":          avg_dist,
            "correct_concepts":  str(correct_concepts),
            "student_concepts":  str(student_concepts),
        })
        prog.progress((i + 1) / total, text=f"Scored {i+1}/{total}")

    st.session_state["results_df"] = pd.DataFrame(rows)
    st.session_state["scored"]     = True
    st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.get("scored"):
    results_df = st.session_state["results_df"]

    st.divider()
    st.header("Results")

    # ── Summary table ────────────────────────────────────────────────────
    summary = (
        results_df.groupby("student_id")
        .agg(
            Questions=("question_id", "count"),
            Correct=("binary", "sum"),
            KG_Score=("kg_score", "mean"),
        )
        .reset_index()
        .rename(columns={"student_id": "Student"})
    )
    summary["Binary Score"] = summary.apply(
        lambda r: f"{int(r.Correct)}/{int(r.Questions)}", axis=1
    )
    summary["KG Score"] = summary["KG_Score"].round(3)
    summary = summary[["Student", "Binary Score", "KG Score"]]

    st.subheader("Summary")
    st.dataframe(summary, width='stretch', hide_index=True)

    # ── Per-student detail ───────────────────────────────────────────────
    st.subheader("Per-Student Detail")

    for student_id in results_df["student_id"].unique():
        s_df   = results_df[results_df["student_id"] == student_id].copy()
        avg_kg = s_df["kg_score"].mean()
        correct= s_df["binary"].sum()
        total  = len(s_df)

        with st.expander(
            f"**{student_id}**  —  Binary: {correct}/{total}  ·  KG Score: {avg_kg:.3f}"
        ):
            display = s_df[
                ["question_id", "correct_answer", "student_answer", "binary", "kg_score", "avg_dist"]
            ].copy()
            display.columns = ["Q", "Correct Answer", "Student Answer", "✓", "KG Score", "Graph Dist"]
            display["✓"] = display["✓"].map({1: "✓", 0: "✗"})
            st.dataframe(display, width='stretch', hide_index=True)

    # ── Download ─────────────────────────────────────────────────────────
    st.divider()
    csv_out = results_df.to_csv(index=False, encoding="utf-8")
    st.download_button(
        "📥 Download Full Results CSV",
        csv_out,
        "kg_scores.csv",
        "text/csv",
        width='stretch',
    )

    if st.button("🔄 Score Again with Different Settings", width='stretch'):
        del st.session_state["scored"]
        del st.session_state["results_df"]
        st.rerun()
