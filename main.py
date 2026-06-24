# main.py
import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    GOOGLE_API_KEY,
    DATA_ROOT,
    MODEL_NAME,
    TEMPERATURE,
    TIMEOUT,
    MAX_RETRIES,
    NUM_SAMPLES,
    SLEEP_SEC,
    MAX_EVIDENCE_TRIPLES
)

from data_loader import load_fb15k237
from degree import make_train_node_degree_df
from multihop import generate_k_hop_path_triples
from llm_runner import run_llm_kgc_on_dataframe


def main():
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

    train_data, valid_data, test_data, train_triples, valid_triples, test_triples = load_fb15k237(
        root=DATA_ROOT
    )

    print("train triples:", len(train_triples))
    print("valid triples:", len(valid_triples))
    print("test triples :", len(test_triples))
    print("num nodes    :", train_data.num_nodes)

    train_node_degree_df = make_train_node_degree_df(
        train_triples=train_triples,
        num_nodes=train_data.num_nodes
    )

    print(train_node_degree_df.head())
    khop_dfs = {}

    for hop in [1, 2, 3, 4]:
        print(f"\nGenerating {hop}-hop triples...")

        df = generate_k_hop_path_triples(
            triples_df=test_triples,
            hop=hop,
            train_node_degree_df=train_node_degree_df
        )

        df = df.sort_values("degree_heterogeneity", ascending=True)
        khop_dfs[hop] = df

        print(f"{hop}-hop rows:", len(df))
        print(df)
        print(df.columns)

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_retries=MAX_RETRIES,
        timeout=TIMEOUT
    )

    result_dfs = []
    question_dfs = []

    for hop in [1, 2, 3, 4]:
        results, questions = run_llm_kgc_on_dataframe(
            df=khop_dfs[hop],
            train_triples=train_triples,
            llm=llm,
            hop_name=f"{hop}hop",
            num_samples=NUM_SAMPLES,
            max_evidence_triples=MAX_EVIDENCE_TRIPLES,
            sleep_sec=SLEEP_SEC
        )

        result_dfs.append(results)
        question_dfs.append(questions)


    all_results = pd.concat(result_dfs, ignore_index=True)
    all_questions = pd.concat(question_dfs, ignore_index=True)

    os.makedirs("results", exist_ok=True)

    all_results.to_csv("results/all_results.csv", index=False)
    all_questions.to_csv("results/all_questions.csv", index=False)

    print("\nDone.")
    print(all_results)


if __name__ == "__main__":
    main()