#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive (HTML) visualization of fused entities (prefix 'set:')
from a SetKnowledgeGraph built from a DAKGEA dataset.
"""

from rdflib import Graph, Literal, URIRef
from pyvis.network import Network


# ==============================================================================
# INTERACTIVE VISUALIZATION FUNCTION
# ==============================================================================

def visualize_fused_entities_html(
    g: Graph,
    output_path: str = "fused_entities.html",
    height: str = "800px",
    width: str = "100%",
    physics: bool = True,
    max_triples: int = 2000,
):
    """
    Display only the merged entities (prefixed with 'set:') and their literals.
    Create placeholders for missing nodes (to avoid PyVis AssertionErrors).
    """
    net = Network(height=height, width=width, directed=True, notebook=False)
    net.barnes_hut()

    def clean_label(x: str) -> str:
        x = x.replace("|", "\n").replace("set:", "set:\n")
        if len(x) > 80:
            x = x[:77] + "..."
        return x

    # --- Filter only relevant triples: subject/object containing 'set:' ---
    fused_triples = []
    for s, p, o in g.triples((None, None, None)):
        if (isinstance(s, URIRef) and str(s).startswith("set:")) or (
            isinstance(o, URIRef) and str(o).startswith("set:")
        ):
            fused_triples.append((s, p, o))
    if not fused_triples:
        print("⚠️ No 'set:' entities found in the graph!")
        return

    # --- If too large, sample ---
    if len(fused_triples) > max_triples:
        import random
        random.seed(42)
        fused_triples = random.sample(fused_triples, max_triples)
        print(f"⚠️ Too many fused nodes: displaying a subset of {len(fused_triples)} triples.")

    node_map = net.node_map

    for s, p, o in fused_triples:
        s_str, p_str, o_str = str(s), str(p), str(o)
        s_label, p_label, o_label = clean_label(s_str), clean_label(p_str), clean_label(o_str)

        # --- Source node ---
        if s_str not in node_map:
            color = "#264653" if s_str.startswith("set:") else "#b0b0b0"
            shape = "ellipse" if s_str.startswith("set:") else "dot"
            size = 25 if s_str.startswith("set:") else 10
            net.add_node(s_str, label=s_label, color=color, shape=shape, size=size)

        # --- Target node ---
        if o_str not in node_map:
            if isinstance(o, Literal):
                net.add_node(o_str, label=o_label, color="#f4a261", shape="diamond", size=12)
            elif isinstance(o, URIRef) and o_str.startswith("set:"):
                net.add_node(o_str, label=o_label, color="#2a9d8f", shape="ellipse", size=25)
            else:
                # Placeholder (non-fused URI)
                net.add_node(o_str, label=o_label, color="#d3d3d3", shape="dot", size=8)

        # --- Labeled edge ---
        net.add_edge(s_str, o_str, label=p_label, color="#1d3557")

    # --- Layout and interaction ---
    net.toggle_physics(physics)
    net.show_buttons(filter_=["physics"])

    # --- HTML export compatible with PyCharm / Python 3.13 ---
    try:
        net.write_html(output_path, open_browser=False, notebook=False)
    except Exception as e:
        print(f"⚠️ PyVis.show() error: {e}")
        html = net.generate_html(notebook=False)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    print(f"✅ Fused entities graph exported to: {output_path}")


# ==============================================================================
# MAIN SCRIPT
# ==============================================================================

if __name__ == "__main__":
    from src.core.dataset.reader.dataset_reader_factory import DatasetReaderFactory
    from src.augmentation.methods.plm.set_knowledge_graph.set_knowledge_graph import SetKnowledgeGraph
    from src.reduction.registry import load_builtin_reducers, REDUCTION_REGISTRY

    # ----------------------------------------------------------------------------
    # 1. Dataset loading
    # ----------------------------------------------------------------------------
    reader = DatasetReaderFactory.create_reader("bert_int")
    dataset = reader.read("/home/federico/Programming/Python/DAKGEA/data/raw/openea/BBC_DB/attribute_data")

    # ----------------------------------------------------------------------------
    # 2. Reduction (optional)
    # ----------------------------------------------------------------------------
    load_builtin_reducers()
    reducer = REDUCTION_REGISTRY.get("random_entities")(
        {"reduction": {"target_entities": 400}, "experiment": {"seed": 11037}}
    )
    reducer.reduce(dataset)

    # ----------------------------------------------------------------------------
    # 3. SetKnowledgeGraph creation
    # ----------------------------------------------------------------------------
    skg = SetKnowledgeGraph.from_dataset(dataset)
    print(skg.summary())

    print("------------------------------------")
    for i, (s, p, o) in enumerate(skg.triples((None, None, None))):
        print(f"<{s}> <{p}> \"{str(o).strip()}\"")
        if i > 10:
            print("...")
            break

    # ----------------------------------------------------------------------------
    # 4. Display only fused entities
    # ----------------------------------------------------------------------------
    visualize_fused_entities_html(skg, "fused_entities.html")

    print("\n🌐 Open 'fused_entities.html' in the browser to explore the fused entities!")
