"""
Step 2: Markdown → Knowledge Graph
Extracts entities and relationships chunk-by-chunk via LLM,
builds a NetworkX graph, and exports:
  - an interactive HTML visualization
  - a JSON snapshot of all nodes and edges

Install: pip install openai networkx pyvis
"""

import json
import re
import sys
import argparse
from pathlib import Path

import networkx as nx
from llm_client import get_client, call_llm_json, DEFAULT_MODEL


# ---------------------------------------------------------------------------
# 1. Markdown chunking
# ---------------------------------------------------------------------------

def chunk_markdown(md_text: str, max_chars: int = 3000) -> list[str]:
    """Split markdown into chunks at heading boundaries, respecting max_chars."""
    heading_pattern = re.compile(r"^#{1,3} .+", re.MULTILINE)
    splits = [m.start() for m in heading_pattern.finditer(md_text)]

    if not splits:
        paragraphs = [p.strip() for p in md_text.split("\n\n") if p.strip()]
        chunks, current = [], ""
        for p in paragraphs:
            if len(current) + len(p) > max_chars and current:
                chunks.append(current.strip())
                current = p
            else:
                current += "\n\n" + p
        if current:
            chunks.append(current.strip())
        return chunks

    chunks = []
    for i, start in enumerate(splits):
        end = splits[i + 1] if i + 1 < len(splits) else len(md_text)
        section = md_text[start:end].strip()
        if len(section) > max_chars:
            parts = section.split("\n\n")
            current = ""
            for part in parts:
                if len(current) + len(part) > max_chars and current:
                    chunks.append(current.strip())
                    current = part
                else:
                    current += "\n\n" + part
            if current:
                chunks.append(current.strip())
        else:
            chunks.append(section)
    return chunks


# ---------------------------------------------------------------------------
# 2. Entity / relationship extraction via LLM
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """You are an expert educational ontologist building a pedagogical knowledge graph from a cognitive psychology textbook.
Your goal is to map the core academic concepts, their taxonomies, and their criterial features. Do not extract trivial grammar links or passing examples (e.g., skip "cats" or "Victorian houses").

You are restricted to the following Pedagogical Relationship types:
1. "IS_A_TYPE_OF" (Taxonomy/Hierarchy: Concept A is a specific sub-category of Concept B)
2. "CONTRASTS_WITH" (Dichotomy: Concept A is presented as the academic opposite or alternative to Concept B)
3. "HAS_CRITERIAL_FEATURE" (Definition: Concept B is a defining, essential attribute of Concept A)
4. "STUDIED_VIA" (Methodology: Concept A is investigated using Concept B)
5. "REPRESENTS" (Symbolism: Concept A stands in for or models Concept B)

Return ONLY valid JSON in this exact format:
{
  "entities": [
    {"id": "unique_snake_case_id", "label": "Display Name", "type": "Theory|Concept|Methodology|Category"}
  ],
  "relationships": [
    {"source": "entity_id", "target": "entity_id", "relation": "EXACT_RELATION_FROM_LIST"}
  ]
}

Rules:
- Extract only core academic entities.
- "relation" MUST be exactly one of the 5 allowed relationship types.
- If no academic relationships exist, return empty arrays.
"""

def extract_graph_from_chunk(client, chunk: str, model: str) -> dict:
    try:
        return call_llm_json(client, EXTRACTION_SYSTEM, chunk, model=model)
    except (json.JSONDecodeError, Exception) as e:
        print(f"  [warn] Skipping chunk: {e}")
        return {"entities": [], "relationships": []}


# ---------------------------------------------------------------------------
# 3. Build NetworkX graph
# ---------------------------------------------------------------------------

def build_graph(all_data: list[dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    for data in all_data:
        for ent in data.get("entities", []):
            node_id = ent["id"]
            if not G.has_node(node_id):
                G.add_node(node_id, label=ent.get("label", node_id), type=ent.get("type", "Other"))
        for rel in data.get("relationships", []):
            src, tgt = rel.get("source"), rel.get("target")
            if src and tgt and G.has_node(src) and G.has_node(tgt):
                if G.has_edge(src, tgt):
                    G[src][tgt]["relation"] += f" / {rel['relation']}"
                else:
                    G.add_edge(src, tgt, relation=rel.get("relation", ""))
    return G


# ---------------------------------------------------------------------------
# 4. Pyvis interactive visualization
# ---------------------------------------------------------------------------

TYPE_COLORS = {
    "Theory":      "#9467bd",
    "Concept":     "#f28e2b",
    "Methodology": "#2ca02c",
    "Category":    "#1f77b4",
    "Other":       "#b07aa1",
}


def visualize_graph(G: nx.DiGraph, output_html: Path) -> None:
    try:
        from pyvis.network import Network
    except ImportError:
        print("pyvis not installed - skipping HTML visualization. Run: pip install pyvis")
        return

    net = Network(height="800px", width="100%", directed=True, notebook=False)
    net.set_options("""{
      "physics": {
        "barnesHut": { "gravitationalConstant": -8000, "springLength": 150 },
        "stabilization": { "iterations": 200 }
      },
      "edges": { "arrows": { "to": { "enabled": true } }, "font": { "size": 10 } },
      "nodes": { "font": { "size": 13 } }
    }""")

    for node_id, attrs in G.nodes(data=True):
        label = attrs.get("label", node_id)
        node_type = attrs.get("type", "Other")
        color = TYPE_COLORS.get(node_type, "#b07aa1")
        net.add_node(node_id, label=label, title=f"{label}\n({node_type})", color=color, size=20)

    for src, tgt, attrs in G.edges(data=True):
        relation = attrs.get("relation", "")
        net.add_edge(src, tgt, title=relation, label=relation)

    net.save_graph(str(output_html))
    print(f"Visualization: {output_html}")


# ---------------------------------------------------------------------------
# 5. JSON export
# ---------------------------------------------------------------------------

def export_json(G: nx.DiGraph, output_json: Path) -> None:
    data = {
        "nodes": [{"id": n, **attrs} for n, attrs in G.nodes(data=True)],
        "edges": [{"source": u, "target": v, **attrs} for u, v, attrs in G.edges(data=True)],
    }
    output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Graph JSON:    {output_json}")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build a knowledge graph from a Markdown file")
    parser.add_argument("md", help="Path to the input .md file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model to use")
    parser.add_argument("--chunk-size", type=int, default=3000, help="Max chars per chunk")
    parser.add_argument("--out-dir", help="Output directory (default: same as input file)")
    args = parser.parse_args()

    md_path = Path(args.md)
    if not md_path.exists():
        sys.exit(f"File not found: {md_path}")

    out_dir = Path(args.out_dir) if args.out_dir else md_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = md_path.stem

    client = get_client()
    md_text = md_path.read_text(encoding="utf-8")
    chunks = chunk_markdown(md_text, max_chars=args.chunk_size)
    print(f"Chunks: {len(chunks)}  |  Model: {args.model}")

    all_data = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}/{len(chunks)} ({len(chunk)} chars)...", end=" ", flush=True)
        result = extract_graph_from_chunk(client, chunk, args.model)
        all_data.append(result)
        print(f"{len(result.get('entities', []))} entities, {len(result.get('relationships', []))} relations")

    G = build_graph(all_data)
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    export_json(G, out_dir / f"{stem}_kg.json")
    visualize_graph(G, out_dir / f"{stem}_kg.html")
    print("Done.")


if __name__ == "__main__":
    main()
