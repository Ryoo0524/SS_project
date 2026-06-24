import pandas as pd
from graph_utils import build_directed_adjacency, get_k_hop_degree_stats_from_adj

def generate_k_hop_path_triples(
        triples_df,
        hop,
        train_node_degree_df=None,
        degree_col="total_degree",
        simple_path=True,
        max_paths=None,
        max_paths_per_head=None,
        missing_degree_value=0
):
    """
    test set을 변형해서 만든 멀티홉 데이터프레임을 반환한다.
    데이터프레임은 head, relation_path, tail, hop, node_path 컬럼이 있다.
    """
    assert hop >= 1

    adj = build_directed_adjacency(triples_df)
    results = []

    if train_node_degree_df is not None:
        degree_map = (
            train_node_degree_df
            .set_index("node")[degree_col]
            .to_dict()
        )
    else:
        degree_map = None

    start_nodes = sorted(adj.keys())

    for h in start_nodes:
        per_head_count = 0

        stack = [(h, [], [h])]

        while stack:
            cur_node, rel_path, node_path = stack.pop()

            if len(rel_path) == hop:
                row = {
                    "head": h,
                    "relation_path": tuple(rel_path),
                    "tail": cur_node,
                    "hop": hop,
                    "node_path": tuple(node_path)
                }

                if degree_map is not None:
                    node_degrees = [
                        int(degree_map.get(int(node), missing_degree_value))
                        for node in node_path
                    ]

                    row["node_degrees"] = tuple(node_degrees)
                    row["path_degree_sum"] = int(sum(node_degrees))

                degree_stats = get_k_hop_degree_stats_from_adj(h, adj, hop, degree_map,
                                                                        missing_degree_value=0,
                                                                        include_start_node=True)

                row["degree_heterogeneity"] = degree_stats["degree_heterogeneity"]
                row["target_nodes"] = degree_stats["target_nodes"]
                results.append(row)

                per_head_count += 1

                if max_paths is not None and len(results) >= max_paths:
                    return pd.DataFrame(results)

                if max_paths_per_head is not None and per_head_count >= max_paths_per_head:
                    break

                continue

            for r_next, next_node in adj.get(cur_node, []):
                if simple_path and next_node in node_path:
                    continue

                stack.append((
                    next_node,
                    rel_path + [r_next],
                    node_path + [next_node]
                ))

    return pd.DataFrame(results)