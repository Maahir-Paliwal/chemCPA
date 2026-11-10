from pathlib import Path
from typing import List
import pandas as pd
import torch
import numpy as np

from chemCPA.paths import EMBEDDING_DIR


def _check_smiles_availability(smiles: List[str], df: pd.DataFrame) -> tuple[set, set, list]:
    """
    Return the embedding of each SMILES in the order given.
    Raises a ValueError if the table has duplicate SMILES or lacks a requested SMILES. 

    Args:
        smiles: List of SMILES string to look up
        df: Dataframe with embeddings and SMILES, containing SMILES as the index
    
    Returns:
        tuple containing:
            1. set of present SMILES
            2. set of missing SMILES (always empty)
            3. List of embeddings
    """
    if df.index.has_duplicates:
        duplicates = df.index[df.index.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate SMILES found in the embedding table: {duplicates}")

    missing_smiles = set(smiles) - set(df.index)
    if missing_smiles:
        raise ValueError(f"Missing SMILES in the embedding table: {missing_smiles}")

    emb_list = [df.loc[smile].values for smile in smiles]
    return set(df.index), missing_smiles, emb_list
    


def get_chemical_representation(
    smiles: List[str],
    embedding_model: str,
    data_path=None,
    device="cuda",
):
    """
    Given a list of SMILES strings, returns the embeddings produced by the embedding model.
    The embeddings are loaded from disk without ever running the embedding model.

    :return: torch.nn.Embedding, shape [len(smiles), dim_embedding]. Embeddings are ordered as in `smiles`-list.
    """
    print("Looking for the following number of smiles:", len(smiles))
    if data_path is None:
        assert embedding_model in (
            "grover_base",
            "weave",
            "MPNN",
            "AttentiveFP",
            "GCN",
            "seq2seq",
            "rdkit",
            "jtvae",
            "zeros",
            "chemvae",
        )
        data_path = Path(EMBEDDING_DIR)
        assert Path(data_path).exists(), f"{data_path} does not exist."

        df = None
        if embedding_model == "grover_base":
            df = pd.read_parquet(data_path / "grover" / "data" / "embeddings" / "grover_base.parquet")
        elif embedding_model == "weave":
            df = pd.read_parquet(
                data_path / "dgl" / "data" / "embeddings" / "Weave_canonical_PCBA_embedding_lincs_trapnell.parquet"
            )
        elif embedding_model == "MPNN":
            df = pd.read_parquet(
                data_path / "dgl" / "data" / "embeddings" / "MPNN_canonical_PCBA_embedding_lincs_trapnell.parquet"
            )
        elif embedding_model == "GCN":
            df = pd.read_parquet(
                data_path / "dgl" / "data" / "embeddings" / "GCN_canonical_PCBA_embedding_lincs_trapnell.parquet"
            )
        elif embedding_model == "AttentiveFP":
            df = pd.read_parquet(
                data_path
                / "dgl"
                / "data"
                / "embeddings"
                / "AttentiveFP_canonical_PCBA_embedding_lincs_trapnell.parquet"
            )
        elif embedding_model == "seq2seq":
            df = pd.read_parquet(Path(data_path) / "seq2seq" / "data" / "seq2seq.parquet")
        elif embedding_model == "rdkit":
            df_path = data_path / "rdkit" / "data" / "embeddings" / "embeddings/rdkit/my_lincs_embeddings.parquet"
            print(df_path)
            df = pd.read_parquet(df_path)
            print(f"Total SMILES in the embedding file: {len(df)}")
        elif embedding_model == "jtvae":
            df = pd.read_parquet(data_path / "jtvae" / "data" / "jtvae_dgl.parquet")
        elif embedding_model == "chemvae":
            df = pd.read_parquet(data_path / "chemvae" / "chemvae.parquet")

        if df is not None:
            _, _, emb_list = _check_smiles_availability(smiles, df)
            emb = torch.tensor(np.stack(emb_list), dtype=torch.float32, device=device)
            assert emb.shape[0] == len(smiles)
        else:
            assert embedding_model == "zeros"
            emb = torch.zeros((len(smiles), 256))
    else:
        print(f"Loading embeddings from {data_path}")
        data_path = Path(data_path)
        assert data_path.exists(), f"{data_path} does not exist."
        df = pd.read_parquet(data_path)
        _, _, emb_list = _check_smiles_availability(smiles, df)
        emb = torch.tensor(np.stack(emb_list), dtype=torch.float32, device=device)
        assert emb.shape[0] == len(smiles)

    return torch.nn.Embedding.from_pretrained(emb, freeze=True)
