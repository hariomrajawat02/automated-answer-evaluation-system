import spacy
import networkx as nx
from sentence_transformers import SentenceTransformer, util

# ----------------------------
# Load models
# ----------------------------
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Text → Graph (SVO extraction)
# ----------------------------
def text_to_graph(text):
    doc = nlp(text)
    G = nx.DiGraph()

    for token in doc:
        # subject
        if token.dep_ == "nsubj":
            subject = token.text.lower()
            verb = token.head.lemma_.lower()

            # object
            for child in token.head.children:
                if child.dep_ in ("dobj", "pobj", "attr"):
                    obj = child.text.lower()
                    G.add_edge(subject, obj, relation=verb)

    return G

# ----------------------------
# Semantic similarity between two words/phrases
# ----------------------------
def semantic_similarity(a, b):
    emb1 = embedder.encode(a, convert_to_tensor=True)
    emb2 = embedder.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

# ----------------------------
# Node matching (semantic)
# ----------------------------
def match_nodes(key_nodes, student_nodes, threshold=0.6):
    matched = {}
    missing = []

    for kn in key_nodes:
        best_match = None
        best_score = 0

        for sn in student_nodes:
            score = semantic_similarity(kn, sn)
            if score > best_score:
                best_score = score
                best_match = sn

        if best_score >= threshold:
            matched[kn] = best_match
        else:
            missing.append(kn)

    return matched, missing

# ----------------------------
# Edge comparison (relations)
# ----------------------------
def match_edges(G_key, G_student, node_mapping):
    matched_edges = []
    missing_edges = []

    for u, v, data in G_key.edges(data=True):
        rel = data["relation"]

        if u in node_mapping and v in node_mapping:
            su = node_mapping[u]
            sv = node_mapping[v]

            if G_student.has_edge(su, sv):
                matched_edges.append((u, v))
            else:
                missing_edges.append((u, v))
        else:
            missing_edges.append((u, v))

    return matched_edges, missing_edges

# ----------------------------
# Final evaluation
# ----------------------------
def evaluate_answer(key_text, student_text):
    G_key = text_to_graph(key_text)
    G_student = text_to_graph(student_text)

    key_nodes = list(G_key.nodes())
    student_nodes = list(G_student.nodes())

    # Node matching
    node_map, missing_nodes = match_nodes(key_nodes, student_nodes)

    # Edge matching
    matched_edges, missing_edges = match_edges(G_key, G_student, node_map)

    # Scores
    concept_score = len(node_map) / len(key_nodes) if key_nodes else 0
    relation_score = len(matched_edges) / len(G_key.edges()) if G_key.edges() else 0

    # Semantic similarity (overall text)
    overall_sim = semantic_similarity(key_text, student_text)

    # Final score (7 marks)
    final_score = (
        4 * concept_score +
        2 * relation_score +
        1 * overall_sim
    )

    # Feedback
    feedback = {
        "matched_concepts": node_map,
        "missing_concepts": missing_nodes,
        "missing_relations": missing_edges
    }

    return round(final_score, 2), feedback

key_answer = """
TCP is a connection oriented protocol that ensures reliable data transmission.
It uses acknowledgments and retransmission.
"""

student_answer = """
TCP is a protocol that provides reliable communication.
It uses acknowledgements.
"""

score, feedback = evaluate_answer(key_answer, student_answer)

print("Score (out of 7):", score)
print("\nFeedback:")
for k, v in feedback.items():
    print(k, ":", v)