import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator, DataStructs
from transformers import AutoTokenizer, AutoModel

# ----------------------------------------------------------------------------
# 0) File paths and model identifiers
# ----------------------------------------------------------------------------
DRUG_INFO_CSV      = "drug_info.csv"            # maps drug names to DrugBank IDs
DRUG_SMILES_CSV    = "DrugBankID2SMILES.csv"   # maps DrugBank IDs to SMILES
MODEL_FILE         = "lgbm_full.booster"        # your saved LightGBM booster
CHEMBERTA_ID       = "seyonec/ChemBERTa-zinc-base-v1"

# ----------------------------------------------------------------------------
# 1) Load mappings: names → IDs, IDs → SMILES
# ----------------------------------------------------------------------------
# Load drug_info.csv and extract name_x as the drug name key
drug_info_df = pd.read_csv("drug_info.csv")
drug_info_df["name_x"] = drug_info_df["name_x"].astype(str).str.strip().str.lower()
drug_info_df["drugbank-id"] = drug_info_df["drugbank-id"].astype(str).str.strip()

# Use name_x as keys
drug_name_to_id = dict(zip(drug_info_df["name_x"], drug_info_df["drugbank-id"]))

# You should also ensure drug_to_smiles is similarly consistent
drug_smiles_df = pd.read_csv("DrugBankID2SMILES.csv", dtype=str)
drug_smiles_df.dropna(subset=["drugbank_id", "smiles"], inplace=True)
drug_to_smiles = dict(zip(drug_smiles_df["drugbank_id"], drug_smiles_df["smiles"]))
# ----------------------------------------------------------------------------
# 2) Fingerprint & embedding utilities
# ----------------------------------------------------------------------------
fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def validate_smiles(smiles: str):
    if not isinstance(smiles, str) or pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except:
        return None
    return mol


def get_morgan_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# initialize ChemBERTa tokenizer+model once
tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_ID)
smiles_model = AutoModel.from_pretrained(CHEMBERTA_ID)
smiles_model.eval()

def get_chemberta_embedding(smiles):
    # Example placeholder logic
    if smiles is None:
        return None

    # Assuming you're using a tokenizer and model (e.g., ChemBERTa)
    inputs = tokenizer(smiles, return_tensors="pt")
    outputs = smiles_model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()  # or whatever you expect

    return emb


# ----------------------------------------------------------------------------
# 3) Load LightGBM model
# ----------------------------------------------------------------------------
booster = lgb.Booster(model_file=MODEL_FILE)

# ----------------------------------------------------------------------------
# 4) High‑level function: predict interaction for a list of drug names
# ----------------------------------------------------------------------------
def predict_interaction_for_drugs(drug_names: list[str], threshold: float = 0.5) -> pd.DataFrame:
    """
    Given a list of drug names (case-insensitive), returns a DataFrame with:
      - DrugBankID
      - probability of interaction
      - binary prediction (0=no, 1=yes)
    """
    # 4.1) Map names to IDs
    ids = []
    for name in drug_names:
        key = name.strip().lower()
        if key not in drug_name_to_id:
            raise KeyError(f"Drug name not found: {name}")
        ids.append(drug_name_to_id[key])

    # 4.2) Map IDs to SMILES
    smiles_list = []
    for db_id in ids:
        if db_id not in drug_to_smiles:
            raise KeyError(f"SMILES not found for ID: {db_id}")
        smiles_list.append(drug_to_smiles[db_id])

    # 4.3) Generate embeddings per drug
    embeddings = []
    for smi in smiles_list:
        fp   = get_morgan_fingerprint(smi)
        cemb = get_chemberta_embedding(smi)
        if fp is None or cemb is None:
            raise ValueError(f"Invalid embedding for SMILES: {smi}")
        embeddings.append(np.concatenate([fp, cemb]))

    # 4.4) Mean‑pool to hyperedge embedding
    hyperedge_emb = np.vstack(embeddings).mean(axis=0)[None, :]

    # 4.5) Predict
    probs = booster.predict(hyperedge_emb)
    preds = (probs >= threshold).astype(int)

    return pd.DataFrame({
        "DrugBankIDs": [ids],
        "probability": probs,
        "prediction": preds
    })

# ----------------------------------------------------------------------------
# 5) Example usage
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    test_names = ["aspirin", "ibuprofen"]  # replace with your drug names
    result_df = predict_interaction_for_drugs(test_names)
    print(result_df.to_string(index=False))
