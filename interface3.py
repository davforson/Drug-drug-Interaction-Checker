import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
import ast
from ddi_hyperedge_predictor import (
    get_morgan_fingerprint, 
    get_chemberta_embedding, 
    drug_to_smiles, 
    drug_name_to_id
)
import lightgbm as lgb

# Load ML model
model = lgb.Booster(model_file="lgbm_full.booster")

# Load the interaction database
combined_df = pd.read_csv("combined.csv")
combined_df['DrugBankID'] = combined_df['DrugBankID'].apply(ast.literal_eval)

# Dropdown title
st.title("💊 Drug–Drug Interaction Checker")

# User selection
selected_drugs = st.multiselect("Select drugs by name", options=list(drug_name_to_id.keys()), max_selections=20)
st.session_state['selected_drugs'] = selected_drugs

# Interaction and ML prediction logic
if st.session_state['selected_drugs']:
    input_names = st.session_state['selected_drugs']
    input_ids = []
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

        # ML Prediction
        smiles_list = [drug_to_smiles.get(i) for i in input_ids]

        if all(s is not None for s in smiles_list):
            st.subheader("🤖 Predicted Interaction Likelihoods (ML Model)")
            for (name1, sm1), (name2, sm2) in combinations(zip(input_names, smiles_list), 2):
                fp1 = get_morgan_fingerprint(sm1)
                fp2 = get_morgan_fingerprint(sm2)
                emb1 = get_chemberta_embedding(sm1)
                emb2 = get_chemberta_embedding(sm2)

                if all(x is not None for x in [fp1, fp2, emb1, emb2]):
                    avg_fp = np.mean([fp1, fp2], axis=0)
                    avg_emb = np.mean([emb1, emb2], axis=0)
                    features = np.concatenate([avg_fp, avg_emb]).reshape(1, -1)
                    likelihood = model.predict(features)[0]
                    st.write(f"💡 **{name1} + {name2}** → Likelihood: **{likelihood:.2%}**")
                else:
                    st.warning(f"⚠️ Could not generate all features for {name1} + {name2}")

                
        else:
            st.warning("⚠️ One or more SMILES strings missing.")

        # Known interaction display
        if interaction_found:
            st.error("⚠️ Interaction Found!")
            for report in matching_reports:
                st.write(f"**🆔 Report ID:** {report['report_id']}")
                st.write(f"**⏱️ Time:** {report['time']}")
                st.write(f"**🧬 Hyperedge Label:** {report['hyperedge_label']}")
                st.write("———")
        else:
            st.success("✅ No known interaction found for the entered drug combination.")
    else:
        st.info("No interaction. You need two or more drugs to check for an interaction.")
else:
    st.info("Please enter drug names.")
