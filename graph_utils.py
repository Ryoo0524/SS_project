from collections import defaultdict, deque
import numpy as np

def build_directed_adjacency(triples_df):
    """
    triples_df에서 인접 리스트 생성
    """
    adj = defaultdict(list)

    for h, r, t in triples_df[["head", "relation", "tail"]].itertuples(index=False, name=None):
        h, r, t = int(h), int(r), int(t)
        adj[h].append((r, t))

    return adj

def get_k_hop_degree_stats_from_adj(
        start_node,
        adj,
        k,
        degree_map,
        missing_degree_value=0,
        include_start_node=True
):
    """
    start_node에서 k-hop 이내에 도달 가능한 노드들의
    degree sum과 degree heterogeneity를 반환.

    degree_heterogeneity:
        H = mean(k^2) / mean(k)^2
    """
    start_node = int(start_node)

    visited = {start_node}
    q = deque([(start_node, 0)])

    while q:
        cur_node, dist = q.popleft()

        if dist == k:
            continue

        for _, next_node in adj.get(cur_node, []):
            next_node = int(next_node)

            if next_node not in visited:
                visited.add(next_node)
                q.append((next_node, dist + 1))

    # 시작 노드를 heterogeneity 계산에 포함할지 결정
    if include_start_node:
        target_nodes = visited
    else:
        target_nodes = visited - {start_node}

    degrees = np.array(
        [
            int(degree_map.get(node, missing_degree_value))
            for node in target_nodes
        ],
        dtype=float
    )

    if len(degrees) == 0:
        return {
            "degree_sum": 0,
            "degree_heterogeneity": 0.0
        }

    degree_sum = int(degrees.sum())

    mean_k = degrees.mean()

    if mean_k == 0:
        degree_heterogeneity = 0.0
    else:
        mean_k2 = (degrees ** 2).mean()
        degree_heterogeneity = mean_k2 / (mean_k ** 2)

    return {
        "degree_sum": degree_sum,
        "degree_heterogeneity": float(degree_heterogeneity),
        "target_nodes": target_nodes
    }