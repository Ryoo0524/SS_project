# data_loader.py
import pandas as pd
from torch_geometric.datasets import FB15k_237
import os
import urllib.request

save_dir = "./data/FB15k_237/text_mapping"
os.makedirs(save_dir, exist_ok=True)

urls = {
    "FB15k_mid2name.txt": "https://huggingface.co/datasets/KGraph/FB15k-237/resolve/cfefa1d76b959f63bd4c1b896b1c18c9299e5a7e/FB15k_mid2name.txt",
    "FB15k_mid2description.txt": "https://huggingface.co/datasets/KGraph/FB15k-237/resolve/cfefa1d76b959f63bd4c1b896b1c18c9299e5a7e/FB15k_mid2description.txt",
}

for filename, url in urls.items():
    save_path = os.path.join(save_dir, filename)
    urllib.request.urlretrieve(url, save_path)
    print("downloaded:", save_path)

def pyg_data_to_triples(data, split_name):
    edge_index = data.edge_index.cpu()
    edge_type = data.edge_type.cpu()

    df = pd.DataFrame({
        "head": edge_index[0].numpy(),
        "relation": edge_type.numpy(),
        "tail": edge_index[1].numpy()
    })

    df["split"] = split_name
    return df


def load_fb15k237(root="./data/FB15k_237"):
    train_dataset = FB15k_237(root=root, split="train")
    valid_dataset = FB15k_237(root=root, split="val")
    test_dataset = FB15k_237(root=root, split="test")

    train_data = train_dataset[0]
    valid_data = valid_dataset[0]
    test_data = test_dataset[0]

    train_triples = pyg_data_to_triples(train_data, "train")
    valid_triples = pyg_data_to_triples(valid_data, "valid")
    test_triples = pyg_data_to_triples(test_data, "test")

    return train_data, valid_data, test_data, train_triples, valid_triples, test_triples