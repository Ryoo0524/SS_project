# llm_runner.py
import time
import pandas as pd

from prompt_builder import (
    build_train_by_head,
    build_kgc_prompt,
    parse_predicted_tail,
    normalize_relation_path,
    build_fb15k237_id_mappings_from_raw,
    load_mid2name
)


def run_llm_kgc_on_dataframe(
    df,
    train_triples,
    llm,
    hop_name,
    num_samples=None,
    max_evidence_triples=None,
    sleep_sec=0.0
):

    mid2name = load_mid2name(
        "./data/FB15k_237/text_mapping/FB15k_mid2name.txt"
    )

    id_to_mid, mid_to_id, id_to_relation, relation_to_id = (
        build_fb15k237_id_mappings_from_raw([
            "./data/FB15k_237/raw/train.txt",
            "./data/FB15k_237/raw/valid.txt",
            "./data/FB15k_237/raw/test.txt",
        ])
    )

    train_by_head = build_train_by_head(train_triples)

    if num_samples is not None:
        eval_df = df.head(num_samples).copy()
    else:
        eval_df = df.copy()

    results = []
    questions = []

    for local_i, (idx, row) in enumerate(eval_df.iterrows(), start=1):
        prompt, meta = build_kgc_prompt(
            row=row,
            train_by_head=train_by_head,
            max_evidence_triples=max_evidence_triples,
            id_to_mid=id_to_mid,
            mid2name=mid2name,
            id_to_relation=id_to_relation
        )

        questions.append({
            "hop_name": hop_name,
            "row_index": int(idx),
            "prompt": prompt
        })

        print("approx tokens:", len(prompt) // 4)

        try:
            response = llm.invoke(prompt)
            response_text = response.content

            print(response_text[0]["text"])
            predicted_tail = parse_predicted_tail(response_text[0]["text"])

        except Exception as e:
            response_text = str(e)
            predicted_tail = None

        gold_tail = int(row["tail"])

        result = {
            "hop_name": hop_name,
            "row_index": int(idx),
            "head": int(row["head"]),
            "relation_path": tuple(normalize_relation_path(row["relation_path"])),
            "gold_tail": gold_tail,
            "predicted_tail": predicted_tail,
            "correct": predicted_tail == gold_tail,
            "degree_heterogeneity": float(row["degree_heterogeneity"]),
            "path_degree_sum": int(row["path_degree_sum"]) if "path_degree_sum" in row else None,
            "num_target_nodes": meta["num_target_nodes"],
            "num_evidence_triples": meta["num_evidence_triples"],
            "response_text": response_text
        }
        print(result)

        results.append(result)

        print(
            f"[{hop_name}] {local_i}/{len(eval_df)} | "
            f"head={result['head']} | "
            f"gold={gold_tail} | pred={predicted_tail} | "
            f"correct={result['correct']} | "
            f"evidence={result['num_evidence_triples']}"
        )


        if sleep_sec > 0:
            time.sleep(sleep_sec)

    return pd.DataFrame(results), pd.DataFrame(questions)