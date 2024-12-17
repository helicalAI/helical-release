from helical.models.genept.model import GenePT, GenePTConfig
import hydra
from omegaconf import DictConfig
import anndata as ad
from datasets import load_dataset
from helical.utils import get_anndata_from_hf_dataset
import os

def run():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    genept_config = GenePTConfig(device="cuda")
    genept = GenePT(configurer = genept_config)

    # either load via huggingface
    hf_dataset = load_dataset("helical-ai/yolksac_human",split="train[:5%]", trust_remote_code=True, download_mode="reuse_cache_if_exists")
    ann_data = get_anndata_from_hf_dataset(hf_dataset)

    # or load directly
    # ann_data = ad.read_h5ad("./yolksac_human.h5ad")

    anndata,embeddings = genept.process_data(ann_data[:1000])

    embeddings = genept.get_embeddings(data)

    print(embeddings)

if __name__ == "__main__":
    run()