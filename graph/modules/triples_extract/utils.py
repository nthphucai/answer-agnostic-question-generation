import networkx as nx
from graph.utils.visualize_utils import visualize_graph_from_networks
from torch_geometric.utils import from_networkx


def convert_to_geometric(kg_df, return_df=True):
    G = nx.from_pandas_edgelist(
        df=kg_df, source="source", target="target", edge_attr=True
    )  # create_using=nx.MultiDiGraph())
    data = from_networkx(G)
    data.edge_relation = kg_df["relation"].values
    data.relation = kg_df["target"].values
    if return_df:
        return kg_df, data
    else:
        return kg_df
