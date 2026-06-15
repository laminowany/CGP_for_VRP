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

def export_cgp_to_graphviz(genes, opts, filename, only_active):
    rows = opts.y_dim
    columns = opts.x_dim
    dot = Digraph()
    dot.attr(
        rankdir="LR",
        splines="true",
        nodesep="0.4",
        ranksep="0.8"
    )
    dx = 2.0
    dy = -1.2
    for g in genes:
        if g is None:
            continue
        if only_active and not g.active:
            continue
        col = g.pos // rows
        row = g.pos % rows
        x = col * dx
        y = row * dy
        style = "filled" if g.active else "dashed"

        dot.node(
            str(g.pos),
            label=f"{g.pos}\n{gene_type_name(g.type)}",
            style=style,
            fillcolor=gene_color(g.type),
            pos=f"{x},{y}!",
            pin="true",
            width="1",
            height="1",
            fixedsize="true"
        )
    for g in genes:
        if g is None or (only_active and not g.active):
            continue

        for inp in g.inputs:
            dot.edge(str(inp), str(g.pos))

    dot.render(filename, format="png", cleanup=True)