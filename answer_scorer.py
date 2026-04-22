"""
Step 3: Grade a student answer using graph distance.

Given:
  - A pre-built knowledge graph (JSON from md_to_kg.py)
  - A question
  - A correct answer
  - A student's answer

Outputs a normalized similarity score [0.0 - 1.0]:
  1.0  = conceptually identical to correct answer
  0.0  = completely unrelated

How it works:
  1. LLM extracts key concepts from each answer as graph node IDs
  2. For each concept pair (correct x student), compute weighted shortest path length
  3. Edge weights reflect pedagogical proximity:
       - Taxonomic (is-a, type-of)           weight 1.0  same conceptual family
       - Contrast/differentiation             weight 1.5  same schema, explicitly paired
       - Compositional (part-of, includes)    weight 1.5  structural closeness
       - Functional (influences, uses)        weight 2.0  process link
       - Theoretical (based-on, model-of)     weight 2.5  weaker theoretical link
       - Person->concept (studied, proposed)  weight 3.5  researcher link, breaks shortcuts
  4. Average weighted distance -> normalize against weighted diameter -> similarity

Install: pip install openai networkx
"""

import json
import sys
import argparse
from pathlib import Path

import networkx as nx
from llm_client import get_client, call_llm_json, DEFAULT_MODEL


# ---------------------------------------------------------------------------
# 1. Edge weighting
# ---------------------------------------------------------------------------

def relation_to_weight(relation: str) -> float:
    """
    Map a relation phrase to an edge weight.
    Lower weight = conceptually closer (cheaper path to traverse).

    Pedagogical rationale:
    - A student who says the *contrasting* term (e.g. procedural vs declarative
      knowledge) is demonstrating schema awareness — they are in the right
      conceptual family. Contrast edges stay LOW weight.
    - Researcher->concept edges (studied, proposed, developed) are topological
      shortcuts that don't reflect conceptual closeness. They get HIGH weight.
    """
    r = relation.lower().strip()

    # Taxonomic / is-a  (same conceptual family)
    # HAS_CRITERIAL_FEATURE is a definitional link — treated as taxonomically close
    if any(kw in r for kw in [
        'is_a_type_of', 'has_criterial_feature',
        'is a type', 'is type', 'is a form', 'is form', 'is a kind',
        'is a category', 'is an example', 'is an instance', 'is categorized',
        'is classification', 'is subordinate', 'is superordinate',
        'is prototypical', 'is synonymous', 'are a type', 'are a form',
        'is a typical', 'is an atypical',
    ]):
        return 1.0

    # Contrast / differentiation  (same schema, explicitly paired)
    if any(kw in r for kw in [
        'contrasts_with',
        'contrast', 'differ', 'distinct from', 'differentiated',
        'alternative to', 'departure from', 'not the same',
        'is in debate', 'contradicts', 'challenges exclusivity',
        'is alternative interpretation',
    ]):
        return 1.5

    # Compositional  (part-whole)
    if any(kw in r for kw in [
        'part of', 'consists of', 'is component', 'includes',
        'comprises', 'contains', 'composed of', 'are components',
        'is comprised', 'constitute', 'is core component',
    ]):
        return 1.5

    # Methodological  (STUDIED_VIA: concept investigated using a method)
    # Must be checked before the person->concept branch to avoid mis-matching 'studied'
    if 'studied_via' in r:
        return 2.0

    # Functional  (process / causal link)
    if any(kw in r for kw in [
        'represents',
        'influences', 'facilitates', 'supports', 'affects',
        'uses', 'utilizes', 'involves', 'requires', 'enables',
        'processes', 'encodes', 'stores',
        'organizes', 'activates', 'improves', 'enhances',
        'underlies', 'causes', 'results', 'produces', 'explains',
    ]):
        return 2.0

    # Theoretical  (model / explanation link)
    if any(kw in r for kw in [
        'is based on', 'is a model', 'is a theory', 'is modeled',
        'predicts', 'is a mechanism', 'provides architecture',
        'proposes theory', 'is theory of',
    ]):
        return 2.5

    # Person -> concept  (researcher links — inflate to break hub shortcuts)
    if any(kw in r for kw in [
        'studied', 'proposed', 'investigated', 'developed',
        'researched', 'conducted', 'created', 'authored',
        'examined', 'suggested', 'defined', 'introduced',
        'demonstrated', 'collaborated', 'performed', 'tested',
        'co-authored', 'published',
    ]):
        return 3.5

    # Default: general association
    return 2.0


# ---------------------------------------------------------------------------
# 2. Load graph
# ---------------------------------------------------------------------------

def load_graph(json_path: Path) -> nx.Graph:
    """Load the KG JSON and return a weighted undirected graph."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    G = nx.Graph()
    for node in data["nodes"]:
        G.add_node(node["id"], label=node.get("label", node["id"]))
    for edge in data["edges"]:
        relation = edge.get("relation", "")
        weight = relation_to_weight(relation)
        G.add_edge(edge["source"], edge["target"], weight=weight, relation=relation)
    return G


# ---------------------------------------------------------------------------
# 3. Concept extraction
# ---------------------------------------------------------------------------

CONCEPT_SYSTEM = """You are a concept mapper for a knowledge graph.

You will receive:
- A list of node labels that exist in the knowledge graph (the vocabulary)
- An answer text to map

Your job: identify which graph node IDs best represent the key concepts EXPLICITLY stated in the answer text.
Return ONLY valid JSON -- a list of node IDs from the vocabulary.
Example: ["working_memory", "cognitive_load", "attention"]

Rules:
- Only return node IDs that appear in the vocabulary list.
- Map ONLY what the answer explicitly states. Do NOT infer, correct, or guess what the answer should be.
- Return 1 to 5 nodes that most directly match the answer's literal content.
- If no nodes match, return an empty list [].
"""


def extract_concepts(
    client,
    model: str,
    question: str,
    answer: str,
    node_vocabulary: list[dict],  # [{"id": ..., "label": ...}, ...]
) -> list[str]:
    vocab_str = "\n".join(f'- {n["id"]}: {n["label"]}' for n in node_vocabulary)
    user_msg = (
        f"Graph vocabulary:\n{vocab_str}\n\n"
        f"Answer to map: {answer}"
    )
    try:
        result = call_llm_json(client, CONCEPT_SYSTEM, user_msg, model=model)
        if isinstance(result, list):
            return [r for r in result if isinstance(r, str)]
        return []
    except Exception as e:
        print(f"  [warn] Concept extraction failed: {e}")
        return []


# ---------------------------------------------------------------------------
# 4. Weighted distance -> normalized similarity
# ---------------------------------------------------------------------------

def pairwise_distances(G: nx.Graph, sources: list[str], targets: list[str]) -> list[float]:
    """Return weighted shortest-path distances for all (source, target) pairs."""
    distances = []
    for s in sources:
        for t in targets:
            if not G.has_node(s) or not G.has_node(t):
                continue
            if s == t:
                distances.append(0.0)
            elif nx.has_path(G, s, t):
                distances.append(
                    nx.shortest_path_length(G, s, t, weight="weight")
                )
            # Disconnected components: skip
    return distances


def estimate_diameter(G: nx.Graph, sample_size: int = 300) -> float:
    """
    Estimate weighted diameter using sampled Dijkstra.
    Returns a float since weighted distances are not integers.
    """
    if G.number_of_nodes() == 0:
        return 1.0
    largest_cc = max(nx.connected_components(G), key=len)
    subG = G.subgraph(largest_cc)

    import random
    nodes = list(subG.nodes())
    sample = random.sample(nodes, min(sample_size, len(nodes)))

    max_dist = 0.0
    for n in sample:
        lengths = nx.single_source_dijkstra_path_length(subG, n, weight="weight")
        local_max = max(lengths.values())
        if local_max > max_dist:
            max_dist = local_max
    return max_dist


def compute_similarity(
    G: nx.Graph,
    correct_concepts: list[str],
    student_concepts: list[str],
    max_dist: float = 10.0,
    power: float = 1.3,
) -> float:
    """
    Similarity in [0.0, 1.0].

    Formula: max(0, 1 - (avg_dist / max_dist) ^ power)

    - max_dist: reference "maximum meaningful distance" (not graph diameter).
                Increase to be more lenient, decrease to be harsher.
    - power:    controls penalty growth rate.
                1.0 = linear, >1.0 = exponential (farther answers penalized more steeply).
    """
    if not correct_concepts or not student_concepts:
        return 0.0

    distances = pairwise_distances(G, correct_concepts, student_concepts)

    if not distances:
        return 0.0

    avg_dist = sum(distances) / len(distances)
    normalized_dist = min(avg_dist / max_dist, 1.0)
    return round(max(0.0, 1.0 - normalized_dist ** power), 4)


# ---------------------------------------------------------------------------
# 5. Main scoring function
# ---------------------------------------------------------------------------

def score_answer(
    graph_json: str,
    question: str,
    correct_answer: str,
    student_answer: str,
    model: str = DEFAULT_MODEL,
    max_dist: float = 10.0,
    power: float = 1.3,
) -> dict:
    """
    Score a student answer against a correct answer using weighted graph distance.

    Returns a dict with:
      - similarity: float [0.0, 1.0]
      - correct_concepts: list[str]   -- nodes mapped from the correct answer
      - student_concepts: list[str]   -- nodes mapped from the student answer
      - avg_distance: float           -- raw average weighted graph distance
      - max_dist: float               -- reference distance used for normalization
      - power: float                  -- penalty exponent used
    """
    G = load_graph(Path(graph_json))
    node_vocabulary = [
        {"id": n, "label": data.get("label", n)}
        for n, data in G.nodes(data=True)
    ]

    client = get_client()

    print("Extracting concepts from correct answer...")
    correct_concepts = extract_concepts(client, model, question, correct_answer, node_vocabulary)
    print(f"  -> {correct_concepts}")

    print("Extracting concepts from student answer...")
    student_concepts = extract_concepts(client, model, question, student_answer, node_vocabulary)
    print(f"  -> {student_concepts}")

    distances = pairwise_distances(G, correct_concepts, student_concepts)
    avg_distance = round(sum(distances) / len(distances), 4) if distances else None
    similarity = compute_similarity(G, correct_concepts, student_concepts, max_dist, power)

    return {
        "similarity": similarity,
        "correct_concepts": correct_concepts,
        "student_concepts": student_concepts,
        "avg_distance": avg_distance,
        "max_dist": max_dist,
        "power": power,
    }


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score a student answer using weighted knowledge graph distance"
    )
    parser.add_argument("graph_json", help="Path to the _kg.json file produced by md_to_kg.py")
    parser.add_argument("--question", required=True, help="The question being answered")
    parser.add_argument("--correct", required=True, help="The correct answer text")
    parser.add_argument("--student", required=True, help="The student's answer text")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model to use")
    parser.add_argument("--max-dist", type=float, default=10.0,
                        help="Reference distance for normalization (default: 10.0). "
                             "Decrease to penalize more harshly, increase to be more lenient.")
    parser.add_argument("--power", type=float, default=1.3,
                        help="Penalty exponent (default: 1.3). "
                             "1.0=linear, >1.0=exponential growth in penalty with distance.")
    args = parser.parse_args()

    result = score_answer(
        graph_json=args.graph_json,
        question=args.question,
        correct_answer=args.correct,
        student_answer=args.student,
        model=args.model,
        max_dist=args.max_dist,
        power=args.power,
    )

    print("\n" + "=" * 50)
    print(f"  Similarity score : {result['similarity']:.4f}  (0=wrong, 1=correct)")
    print(f"  Avg graph dist   : {result['avg_distance']}")
    print(f"  Max dist (ref)   : {result['max_dist']}")
    print(f"  Power            : {result['power']}")
    print(f"  Correct concepts : {result['correct_concepts']}")
    print(f"  Student concepts : {result['student_concepts']}")
    print("=" * 50)

    out = Path(args.graph_json).parent / "score_result.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Result saved to  : {out}")


if __name__ == "__main__":
    main()
