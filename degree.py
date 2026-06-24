# degree.py
import torch
import pandas as pd

def make_train_node_degree_df(train_triples, num_nodes=None):
    """
    train 데이터셋에 있는 모든 노드들의 차수가 적혀있는 데이터프레임 만들기
    """
    train_nodes = sorted(
        set(train_triples["head"].tolist()) | set(train_triples["tail"].tolist())
    )

    #num_nodes가 없으면 train에 등장한 최대 id 기준으로 노드 갯수를 설정
    if num_nodes is None:
        num_nodes = max(train_nodes) + 1

    in_degree = torch.zeros(num_nodes, dtype=torch.long)
    out_degree = torch.zeros(num_nodes, dtype=torch.long)

    train_heads = torch.tensor(train_triples["head"].values, dtype=torch.long)
    train_tails = torch.tensor(train_triples["tail"].values, dtype=torch.long)

    out_degree.scatter_add_(
        0,
        train_heads,
        torch.ones_like(train_heads)
    )

    in_degree.scatter_add_(
        0,
        train_tails,
        torch.ones_like(train_tails)
    )

    total_degree = in_degree + out_degree

    rows = []

    for node in train_nodes:
        node = int(node)

        rows.append({
            "node": node,
            "in_degree": int(in_degree[node]),
            "out_degree": int(out_degree[node]),
            "total_degree": int(total_degree[node]),
        })

    train_node_degree_df = pd.DataFrame(rows)

    return train_node_degree_df