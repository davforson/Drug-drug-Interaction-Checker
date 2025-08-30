import streamlit as st
import pandas as pd
import ast
import lightgbm as lgb
import numpy as np
import torch
from ddi_hyperedge_predictor import (
    get_morgan_fingerprint, get_chemberta_embedding,
    drug_name_to_id, drug_to_smiles
)

# Load model
model = lgb.Booster(model_file="lgbm_full.booster")


# Load drug info and clean names
drug_info_df = pd.read_csv("drug_info.csv")
drug_info_df.iloc[:, 1] = drug_info_df.iloc[:, 1].str.strip().str.lower().str.replace('"', '')
drug_name_to_id = dict(zip(drug_info_df.iloc[:, 1], drug_info_df.iloc[:, 0]))
drug_names = sorted(drug_name_to_id.keys())

# Load interaction data
combined_df = pd.read_csv("combined.csv")
combined_df['DrugBankID'] = combined_df['DrugBankID'].apply(ast.literal_eval)

# Initialize session state
if 'selected_drugs' not in st.session_state:
    st.session_state['selected_drugs'] = []

st.title("💊 Drug–Drug Interaction Checker")

# Dropdown to add drugs one by one
selected_drug = st.selectbox("Type to search and select a drug", options=drug_names)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("➕ Add Drug"):
        if selected_drug not in st.session_state['selected_drugs']:
            if len(st.session_state['selected_drugs']) < 20:
                st.session_state['selected_drugs'].append(selected_drug)
            else:
                st.warning("⚠️ You can only add up to 20 drugs.")
        else:
            st.info(f"{selected_drug} is already in the list.")
with col2:
    if st.button("🗑️ Clear All"):
        st.session_state['selected_drugs'] = []

# Display selected drugs
st.write("### 🧾 Selected Drugs:")
if st.session_state['selected_drugs']:
    st.write(", ".join(st.session_state['selected_drugs']))
else:
    st.info("No drugs added yet.")

# Interaction check
if st.session_state['selected_drugs']:
    input_names = st.session_state['selected_drugs']
    input_ids = []

    # Convert names to DrugBank IDs
    missing = []
    for name in input_names:
        if name in drug_name_to_id:
            input_ids.append(drug_name_to_id[name])
        else:
            missing.append(name)

    if missing:
        st.warning(f"⚠️ Drug(s) not found: {', '.join(missing)}")

    if len(input_ids) >= 2:
        interaction_found = False
        matching_reports = []

        for _, row in combined_df.iterrows():
            if all(drug in row['DrugBankID'] for drug in input_ids):
                interaction_found = True
                matching_reports.append(row)

        # Store for drug info page
        st.session_state['interacting_drugs'] = input_ids
        st.session_state['entered_drugs'] = input_names
        if interaction_found:
            st.error("⚠️ Interaction Found!")
            for report in matching_reports:
                st.write(f"**🆔 Report ID:** {report['report_id']}")
                st.write(f"**🕒 Time:** {report['time']}")
                st.write(f"**🔗 Hyperedge Label:** {report['hyperedge_label']}")
                st.write("——")
        else:
            st.success("✅ No known interaction found for the entered drug combination.")
    else:
        st.info("Please enter at least two drugs to check for interactions.")
