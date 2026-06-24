# prompt_builder.py
import ast
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

def load_mid2name(path):
    """
    FB15k_mid2name.txt 파일 읽기.
    반환:
        mid2name: Freebase MID -> entity name
    """
    mid2name = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) < 2:
                continue

            mid = parts[0]
            name = parts[1]

            mid2name[mid] = name

    return mid2name



def build_fb15k237_id_mappings_from_raw(raw_source):
    """
    PyG Dataset 객체 또는 raw txt 경로를 받아서
    숫자 entity ID -> Freebase MID,
    숫자 relation ID -> relation 문자열
    매핑을 복원한다.

    raw_source로 넣을 수 있는 것:
    1. PyG FB15k_237 dataset 객체
       build_fb15k237_id_mappings_from_raw(train_dataset)

    2. raw 폴더 경로
       build_fb15k237_id_mappings_from_raw("./data/FB15k_237/raw")

    3. train/valid/test 파일 경로 list
       build_fb15k237_id_mappings_from_raw([
           "./data/FB15k_237/raw/train.txt",
           "./data/FB15k_237/raw/valid.txt",
           "./data/FB15k_237/raw/test.txt"
       ])
    """

    # 1. PyG dataset 객체인 경우
    if hasattr(raw_source, "raw_paths"):
        raw_paths = list(raw_source.raw_paths)

    # 2. list/tuple로 파일 경로들을 직접 준 경우
    elif isinstance(raw_source, (list, tuple)):
        raw_paths = [str(Path(p)) for p in raw_source]

    # 3. 문자열 경로인 경우
    else:
        raw_path = Path(raw_source)

        if raw_path.is_dir():
            # FB15k-237 raw 폴더 안의 파일명 후보
            candidates = [
                raw_path / "train.txt",
                raw_path / "valid.txt",
                raw_path / "val.txt",
                raw_path / "test.txt",
            ]

            raw_paths = [str(p) for p in candidates if p.exists()]

            if len(raw_paths) == 0:
                # 혹시 하위 폴더에 raw가 있는 경우까지 검색
                raw_paths = [str(p) for p in raw_path.rglob("*.txt")]

        elif raw_path.is_file():
            raw_paths = [str(raw_path)]

        else:
            raise FileNotFoundError(f"경로를 찾을 수 없음: {raw_source}")

    print("raw paths used:")
    for p in raw_paths:
        print(" -", p)

    node_dict = {}
    rel_dict = {}

    for path in raw_paths:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        for line in lines:
            if not line.strip():
                continue

            parts = line.split("\t")

            if len(parts) != 3:
                continue

            src, rel, dst = parts

            if src not in node_dict:
                node_dict[src] = len(node_dict)

            if dst not in node_dict:
                node_dict[dst] = len(node_dict)

            if rel not in rel_dict:
                rel_dict[rel] = len(rel_dict)

    id_to_mid = {idx: mid for mid, idx in node_dict.items()}
    mid_to_id = node_dict

    id_to_relation = {idx: rel for rel, idx in rel_dict.items()}
    relation_to_id = rel_dict

    return id_to_mid, mid_to_id, id_to_relation, relation_to_id

def entity_id_to_prompt_text(entity_id, id_to_mid=None, mid2name=None):
    """
    숫자 entity ID를 prompt에 넣기 좋은 텍스트로 변환.

    예:
    123 -> Barack Obama [ID: 123, MID: /m/02mjmr]
    """
    entity_id = int(entity_id)

    if id_to_mid is None:
        return f"Entity_{entity_id}"

    mid = id_to_mid.get(entity_id, f"UNKNOWN_MID_{entity_id}")

    if mid2name is None:
        return f"{mid}"

    name = mid2name.get(mid, mid)

    return f"{name}"


def relation_id_to_prompt_text(relation_id, id_to_relation=None):
    """
    숫자 relation ID를 prompt용 텍스트로 변환.
    relation mapping이 없으면 숫자 ID 그대로 사용.
    """
    relation_id = int(relation_id)

    if id_to_relation is None:
        return f"Relation_{relation_id}"

    rel = id_to_relation.get(relation_id, f"UNKNOWN_RELATION_{relation_id}")

    # /people/person/nationality -> people person nationality
    readable = rel.strip("/").replace("/", " ").replace("_", " ")

    return f"{readable}"



def normalize_target_nodes(target_nodes):
    if target_nodes is None:
        return set()

    if isinstance(target_nodes, set):
        return set(map(int, target_nodes))

    if isinstance(target_nodes, (list, tuple, np.ndarray)):
        return set(map(int, target_nodes))

    if isinstance(target_nodes, str):
        try:
            parsed = ast.literal_eval(target_nodes)
            return set(map(int, parsed))
        except Exception:
            return set()

    return set()


def normalize_relation_path(relation_path):
    if isinstance(relation_path, tuple):
        return tuple(map(int, relation_path))

    if isinstance(relation_path, list):
        return tuple(map(int, relation_path))

    if isinstance(relation_path, str):
        try:
            parsed = ast.literal_eval(relation_path)
            if isinstance(parsed, int):
                return (parsed,)
            return tuple(map(int, parsed))
        except Exception:
            if "->" in relation_path:
                return tuple(map(int, relation_path.split("->")))
            return (int(relation_path),)

    return tuple(relation_path)


def build_train_by_head(train_triples):
    train_by_head = {}

    for h, group in train_triples.groupby("head"):
        train_by_head[int(h)] = group[["head", "relation", "tail"]].copy()

    return train_by_head


def get_train_triples_from_target_nodes(target_nodes, train_by_head):
    target_nodes = normalize_target_nodes(target_nodes)
    frames = []

    for node in sorted(target_nodes):
        if node in train_by_head:
            frames.append(train_by_head[node])

    if len(frames) == 0:
        return pd.DataFrame(columns=["head", "relation", "tail"])

    return pd.concat(frames, ignore_index=True)


def format_train_triples_for_prompt(
        evidence_triples,
        max_evidence_triples=None,
        id_to_mid=None,
        mid2name=None,
        id_to_relation=None
):
    if evidence_triples is None or len(evidence_triples) == 0:
        return "No training triples are available."

    if max_evidence_triples is not None:
        evidence_triples = evidence_triples.head(max_evidence_triples)

    lines = []

    for i, row in enumerate(evidence_triples.itertuples(index=False), start=1):
        h = int(row.head)
        r = int(row.relation)
        t = int(row.tail)

        h_text = entity_id_to_prompt_text(h, id_to_mid, mid2name)
        r_text = relation_id_to_prompt_text(r, id_to_relation)
        t_text = entity_id_to_prompt_text(t, id_to_mid, mid2name)

        lines.append(f"{i}. ({h_text}, {r_text}, {t_text})")

    return "\n".join(lines)


def build_kgc_prompt(
        row,
        train_by_head,
        max_evidence_triples=None,
        id_to_mid=None,
        mid2name=None,
        id_to_relation=None
):
    head = int(row["head"])
    gold_tail = int(row["tail"])
    hop = int(row["hop"])

    head_text = entity_id_to_prompt_text(head, id_to_mid, mid2name)
    gold_tail_text = entity_id_to_prompt_text(gold_tail, id_to_mid, mid2name)

    relation_path = normalize_relation_path(row["relation_path"])

    relation_path_text = [
        relation_id_to_prompt_text(r, id_to_relation)
        for r in relation_path
    ]

    relation_path_str = " -> ".join(relation_path_text)

    target_nodes = normalize_target_nodes(row["target_nodes"])

    evidence_triples = get_train_triples_from_target_nodes(
        target_nodes=target_nodes,
        train_by_head=train_by_head
    )

    evidence_text = format_train_triples_for_prompt(
        evidence_triples=evidence_triples,
        max_evidence_triples=max_evidence_triples,
        id_to_mid=id_to_mid,
        mid2name=mid2name,
        id_to_relation=id_to_relation
    )

    prompt = f"""
You are solving a multi-hop knowledge graph completion task.

Entities are written with their name, integer ID, and Freebase MID.
Relations are written with their readable name and relation ID.

Your task:
Given a head entity and a relation path, predict the missing tail entity.

Query:
Head entity: {head_text}
Relation path: {relation_path_str}
Hop length: {hop}

This means:
Starting from the head entity, follow the relation path and predict the final tail entity.

Available training triples:
The following triples are from the training knowledge graph.
Each triple has the form:
(head entity, relation, tail entity)

{evidence_text}

Important rules:
1. Use only the training triples above as graph evidence.
2. Do not use the hidden gold answer.
3. Predict the tail entity for the query using the head entity and relation path.
4. The answer must be one integer entity ID.
5. Return only valid JSON in the following format:

{{
  "predicted_tail": "notebook"
}}
"""

    meta = {
        "head": head,
        "head_text": head_text,
        "gold_tail": gold_tail,
        "gold_tail_text": gold_tail_text,
        "hop": hop,
        "relation_path": relation_path,
        "relation_path_text": relation_path_text,
        "num_target_nodes": len(target_nodes),
        "num_evidence_triples": len(evidence_triples)
    }
    print(prompt)

    return prompt, meta


def parse_predicted_tail(response_text):
    if response_text is None:
        return None

    text = response_text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        obj = json.loads(text)
        if "predicted_tail" in obj:
            return int(obj["predicted_tail"])
    except Exception:
        pass

    match = re.search(r'"?predicted_tail"?\s*:\s*(-?\d+)', text)
    if match:
        return int(match.group(1))

    match = re.search(r"-?\d+", text)
    if match:
        return int(match.group(0))

    return None