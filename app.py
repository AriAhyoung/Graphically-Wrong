"""
Knowledge Graph Answer Scorer — Streamlit Web App
Run: streamlit run app.py
"""

import json
import hashlib
import sys
import tempfile
import io
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
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
        "Max Distance", 1.0, 10.0, 4.0, 0.5,
        help="Reference distance for normalization. Set this close to your graph's typical longest path. Lower = harsher, more spread.",
    )
    power = st.slider(
        "Leniency", 1.0, 5.0, 2.0, 0.1,
        help="1.0 = linear penalty. Higher = exponential — nearby wrong answers keep more credit, distant ones drop sharply.",
    )

    st.divider()
    st.subheader("Score Preview")
    st.caption("Expected scores at current settings:")
    for d, label, is_correct in [
        (0,   "Correct answer",          True),
        (2,   "Same schema (wrong)",     False),
        (3.5, "Related (wrong)",         False),
        (5.5, "Different domain (wrong)",False),
    ]:
        sim = max(0.0, 1.0 - (min(d, max_dist) / max_dist) ** power)
        score = 1.0 if is_correct else sim * 0.7
        filled = int(score * 10)
        bar = "█" * filled + "░" * (10 - filled)
        st.caption(f"`{bar}` **{score:.2f}**  {label}")

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
        '<span style="font-weight:600;font-size:0.95rem;">① Textbook</span><br>'
        '<span style="font-size:0.75rem;color:#6c757d;font-family:monospace;">.pdf or .md</span>',
        unsafe_allow_html=True,
    )
    pdf_file = st.file_uploader(
        "Textbook", type=["pdf", "md"], label_visibility="collapsed", key="pdf_upload"
    )
    if pdf_file:
        file_bytes = pdf_file.read(); pdf_file.seek(0)
        h = _pdf_hash(file_bytes)
        kg_path = _kg_cache_path(h)
        is_md = pdf_file.name.lower().endswith(".md")
        if kg_path.exists():
            kg_info = json.loads(kg_path.read_text(encoding="utf-8"))
            st.success(
                f"✓ KG already built\n\n"
                f"{len(kg_info['nodes'])} nodes · {len(kg_info['edges'])} edges"
            )
        else:
            hint = "No conversion needed — will build graph directly (~2–3 min)" if is_md \
                else "KG will be built on first run (~3–5 min, saved for reuse)"
            st.info(hint)

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
    pdf_file.seek(0); file_bytes = pdf_file.read()
    is_md = pdf_file.name.lower().endswith(".md")
    qa_file.seek(0);  qa_df     = pd.read_csv(qa_file)
    frames = []
    for f in answers_files:
        f.seek(0); frames.append(pd.read_csv(f))
    answers_df = pd.concat(frames, ignore_index=True)

    h       = _pdf_hash(file_bytes)
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
            # ── Step 1: PDF → Markdown (skip if .md uploaded) ──────────────
            if is_md:
                st.write("📝 **Step 1/2** — Markdown file detected, skipping conversion")
                md_text = file_bytes.decode("utf-8")
            else:
                pdf_mb = len(file_bytes) / 1_000_000
                est_pdf_secs = max(10, int(pdf_mb * 15))
                st.write(f"📄 **Step 1/2** — Reading your textbook…  `est. {est_pdf_secs}s`")
                step1_placeholder = st.empty()
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                    f.write(file_bytes)
                    tmp_pdf = Path(f.name)
                tmp_md = tmp_pdf.with_suffix(".md")
                t0 = time.time()
                with st.spinner("Converting PDF to text… 📖"):
                    pdf_to_markdown(str(tmp_pdf), str(tmp_md))
                elapsed_pdf = int(time.time() - t0)
                md_text = tmp_md.read_text(encoding="utf-8")
                tmp_pdf.unlink(); tmp_md.unlink()
                step1_placeholder.success(f"✅ Done in {elapsed_pdf}s")

            # ── Step 2: Markdown → Graph ────────────────────────────────────
            chunks = chunk_markdown(md_text)
            st.write(f"🔗 **Step 2/2** — Building concept graph ({len(chunks)} chunks)…")
            st.html("""
<canvas id="net" width="560" height="140"
  style="display:block;margin:6px auto 0;border-radius:8px;background:#f8f9fb;"></canvas>
<script>
(function(){
  const cv = document.getElementById('net');
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  const N = 22;
  const nodes = Array.from({length: N}, (_, i) => ({
    x: 36 + Math.random() * (W - 72),
    y: 24 + Math.random() * (H - 48),
    r: 3 + Math.random() * 2.5,
    a: 0, born: i * 9
  }));
  const edges = [];
  for (let i = 0; i < N; i++)
    for (let j = i+1; j < N; j++) {
      const dx = nodes[i].x-nodes[j].x, dy = nodes[i].y-nodes[j].y;
      if (Math.sqrt(dx*dx+dy*dy) < 110 && Math.random() > 0.45)
        edges.push({a:i, b:j, alpha:0, born: Math.max(nodes[i].born,nodes[j].born)+10});
    }
  let frame = 0;
  function reset() {
    frame = 0;
    nodes.forEach((n,i) => { n.a=0; n.born=i*9; });
    edges.forEach(e => { e.alpha=0; e.born=Math.max(nodes[e.a].born,nodes[e.b].born)+10; });
  }
  function draw() {
    ctx.clearRect(0, 0, W, H);
    edges.forEach(e => { if (frame>e.born) e.alpha=Math.min(1,e.alpha+0.04); });
    nodes.forEach(n => { if (frame>n.born) n.a=Math.min(1,n.a+0.07); });
    edges.forEach(e => {
      if (!e.alpha) return;
      const a=nodes[e.a], b=nodes[e.b];
      ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y);
      ctx.strokeStyle=`rgba(99,128,199,${e.alpha*0.35})`; ctx.lineWidth=1; ctx.stroke();
    });
    nodes.forEach(n => {
      if (!n.a) return;
      ctx.beginPath(); ctx.arc(n.x,n.y,n.r+4,0,Math.PI*2);
      ctx.fillStyle=`rgba(99,128,199,${n.a*0.12})`; ctx.fill();
      ctx.beginPath(); ctx.arc(n.x,n.y,n.r,0,Math.PI*2);
      ctx.fillStyle=`rgba(74,109,186,${n.a})`; ctx.fill();
    });
    frame++;
    if (nodes.every(n=>n.a>=1) && edges.every(e=>e.alpha>=1))
      setTimeout(reset, 1200);
    requestAnimationFrame(draw);
  }
  draw();
})();
</script>
""")
            prog = st.progress(0, text="Starting…")
            eta_placeholder = st.empty()
            all_data = []
            t0 = time.time()
            for i, chunk in enumerate(chunks):
                result = extract_graph_from_chunk(client, chunk, DEFAULT_MODEL)
                all_data.append(result)
                elapsed = time.time() - t0
                avg = elapsed / (i + 1)
                remaining = avg * (len(chunks) - i - 1)
                mins, secs = divmod(int(remaining), 60)
                eta_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
                el_mins, el_secs = divmod(int(elapsed), 60)
                elapsed_str = f"{el_mins}m {el_secs}s" if el_mins > 0 else f"{el_secs}s"
                msg = GRAPH_MSGS[i % len(GRAPH_MSGS)]
                prog.progress((i + 1) / len(chunks), text=f"{msg}  ({i+1}/{len(chunks)})")
                eta_placeholder.caption(f"⏱ Elapsed: {elapsed_str} · Est. remaining: {eta_str}")

            eta_placeholder.empty()
            G_build = build_graph(all_data)
            export_json(G_build, kg_path)
            total_secs = int(time.time() - t0)
            el_mins, el_secs = divmod(total_secs, 60)
            total_str = f"{el_mins}m {el_secs}s" if el_mins > 0 else f"{el_secs}s"
            status.update(
                label=f"✓ Knowledge graph ready — {G_build.number_of_nodes()} nodes · {G_build.number_of_edges()} edges · built in {total_str}",
                state="complete",
            )

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
        binary = int(str(row.answer).strip().lower() == str(row.correct_answer).strip().lower())

        if binary == 1:
            kg_score, avg_dist, correct_concepts, student_concepts = 1.0, 0.0, [], []
            final_score = 1.0
        else:
            correct_concepts = extract_concepts(client, DEFAULT_MODEL, row.question, row.correct_answer, vocab)
            student_concepts = extract_concepts(client, DEFAULT_MODEL, row.question, row.answer, vocab)
            dists    = pairwise_distances(G, correct_concepts, student_concepts)
            avg_dist = round(sum(dists) / len(dists), 3) if dists else None
            kg_score = compute_similarity(G, correct_concepts, student_concepts, max_dist, power)
            final_score = kg_score * 0.7

        rows.append({
            "student_id":        row.student_id,
            "question_id":       row.question_id,
            "correct_answer":    row.correct_answer,
            "student_answer":    row.answer,
            "binary":            binary,
            "kg_score":          kg_score,
            "final_score":       final_score,
            "avg_dist":          avg_dist,
            "correct_concepts":  str(correct_concepts),
            "student_concepts":  str(student_concepts),
        })
        prog.progress((i + 1) / total, text=f"Scored {i+1}/{total}")

    st.session_state["results_df"] = pd.DataFrame(rows)
    st.session_state["kg_path"]    = str(kg_path)
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
            Final_Score=("final_score", "sum"),
        )
        .reset_index()
        .rename(columns={"student_id": "Student"})
    )
    summary["Binary Score"] = summary.apply(
        lambda r: f"{int(r.Correct)}/{int(r.Questions)}", axis=1
    )
    summary["Partial Score"] = summary.apply(
        lambda r: f"{round(r.Final_Score, 1)}/{int(r.Questions)}", axis=1
    )
    summary = summary[["Student", "Binary Score", "Partial Score"]]

    st.subheader("Summary")
    st.dataframe(summary, width='stretch', hide_index=True)

    # ── Per-student detail ───────────────────────────────────────────────
    st.subheader("Per-Student Detail")

    for student_id in results_df["student_id"].unique():
        s_df   = results_df[results_df["student_id"] == student_id].copy()
        correct= s_df["binary"].sum()
        total  = len(s_df)
        partial = round(s_df["final_score"].sum(), 1)

        with st.expander(
            f"**{student_id}**  —  Binary: {correct}/{total}  ·  Partial: {partial}/{total}"
        ):
            display = s_df[
                ["question_id", "correct_answer", "student_answer", "binary", "final_score", "kg_score", "avg_dist"]
            ].copy()
            display.columns = ["Q", "Correct Answer", "Student Answer", "✓", "Score", "Similarity", "Graph Dist"]
            display["✓"] = display["✓"].map({1: "✓", 0: "✗"})
            display["Score"] = display["Score"].apply(lambda x: f"{x:.2f}")
            display["Similarity"] = display.apply(
                lambda r: "—" if r["✓"] == "✓" else f"{round(r['Similarity']*100)}%", axis=1
            )
            st.dataframe(display, width='stretch', hide_index=True)

    # ── Knowledge Graph Viewer ───────────────────────────────────────────
    st.divider()
    with st.expander("View Knowledge Graph"):
        if st.session_state.get("kg_path"):
            from pyvis.network import Network
            from answer_scorer import load_graph
            G_viz = load_graph(Path(st.session_state["kg_path"]))
            net = Network(height="560px", width="100%", bgcolor="#fafafa", font_color="#222")
            net.barnes_hut(gravity=-8000, spring_length=120, spring_strength=0.04)
            for node, data in G_viz.nodes(data=True):
                net.add_node(node, label=data.get("label", node),
                             color="#4A6DBB", font={"size": 11}, size=14)
            for u, v, data in G_viz.edges(data=True):
                net.add_edge(u, v, title=data.get("relation", ""),
                             color="#aab4cc", width=1)
            components.html(net.generate_html(), height=580, scrolling=False)

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
