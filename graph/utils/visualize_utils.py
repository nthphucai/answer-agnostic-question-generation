import os
import tempfile
from subprocess import Popen
from sys import stderr
from typing import Dict, List

import requests

import matplotlib.pyplot as plt
import networkx as nx


def annotate(document: str, annotators):
    annotators_str = f'"{",".join(annotators)}"'
    request_string = (
        'http://[::]:9001/?properties={"annotators":'
        + annotators_str
        + ',"outputFormat":"json"}'
    )
    response = requests.post(request_string, data={"data": document})
    return response


def visualize_graph_from_corenlp(triples: List[Dict], png_filename):
    graph = list()
    graph.append("digraph {")
    for er in triples:
        graph.append(
            '"{}" -> "{}" [ label="{}" ];'.format(
                er["subject"], er["object"], er["relation"]
            )
        )
    graph.append("}")
    output_dir = os.path.join(".", os.path.dirname(png_filename))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    out_dot = os.path.join(tempfile.gettempdir(), "graph.dot")
    with open(out_dot, "w") as output_file:
        output_file.writelines(graph)

    command = "dot -Tpng {} -o {}".format(out_dot, png_filename)
    dot_process = Popen(command, stdout=stderr, shell=True)
    dot_process.wait()
    assert (
        not dot_process.returncode
    ), "ERROR: Call to dot exited with a non-zero code status."


def visualize_graph_from_networks(kg_df, figsize=(12, 12)):
    """Given Knowledge Graph as DataFrame type with columns as
    ["subject, "object", "relation"] and draw network from networks library.

    Args:
        kg_df (_type_): _description_
        figsize (tuple, optional): _description_. Defaults to (12,12).
    """
    plt.figure(figsize=figsize)

    G = nx.from_pandas_edgelist(
        df=kg_df,
        source="source",
        target="target",
        edge_attr=True,
        create_using=nx.DiGraph(),
    )
    # pos = nx.spring_layout(G)
    pos = nx.shell_layout(G)

    nx.draw_networkx(
        G, with_labels=True, node_color="skyblue", edge_cmap=plt.cm.Blues, pos=pos
    )
    nx.draw_networkx_edge_labels(
        G=G, pos=pos, edge_labels=dict(zip(G.edges, tuple(kg_df["relation"].values)))
    )
    plt.show()
