from graphviz import Digraph


def gene_type_name(gene_type):
    names = {
        1: "Identity",
        2: "Norm",
        3: "Linear",
        4: "Attention",
        5: "Add",
        6: "GELU",
        7: "ReLU",
    }
    return names.get(gene_type, f"Unknown({gene_type})")


def gene_color(gene_type):
    colors = {
        1: "lightgray",
        2: "lightblue",
        3: "orange",
        4: "red",
        5: "green",
        6: "yellow",
        7: "pink",
    }
    return colors.get(gene_type, "white")


def export_cgp_to_graphviz(genes, filename="cgp_graph"):
    dot = Digraph()
    dot.attr(rankdir="LR")

    # nodes
    for g in genes:
        if g is None or not g.active:
            continue

        label = f"{g.pos}\n{gene_type_name(g.type)}"

        dot.node(
            str(g.pos),
            label=label,
            style="filled",
            fillcolor=gene_color(g.type)
        )

    # edges
    for g in genes:
        if g is None or not g.active:
            continue

        for inp in g.inputs:
            dot.edge(str(inp), str(g.pos))

    dot.render(filename, format="png", cleanup=True)