# Improving Medication Safety: Predicting Potential Drug-Drug Interactions Using Machine Learning

This repository contains the final report for our Health Information Systems project at Carnegie Mellon University.  
The project explores how **machine learning** and **decision support systems (DSS)** can be applied to predict and prevent potential **drug-drug interactions (DDIs)** in healthcare settings. 

---

## Motivation  

Adverse drug–drug interactions (DDIs) remain a significant challenge in healthcare, particularly as the prevalence of polypharmacy continues to rise. While existing clinical trials and drug interaction databases capture many common interactions, they often fail to account for:  

- **Rarely prescribed or orphan drugs**, which have limited usage data. 
- **New or emerging drugs**, where real-world evidence is scarce. 
- **Interactions underrepresented in clinical trials**, due to sample size and design constraints.  

These gaps in detection increase the risk of adverse events, especially for vulnerable patient populations. Addressing them requires tools that can go beyond static, rule-based systems and incorporate predictive modeling rooted in molecular properties.  

This project focused on identifying potential DDIs that are often overlooked—particularly for drugs with low usage frequency (including orphan drugs) and for interactions that cannot be fully captured in clinical trials due to inherent limitations.  

Our proposed DDI checker leverages **structure–activity relationships (SARs)** to group drugs with similar molecular characteristics. By extension, if two drugs share comparable SARs and active moieties at the molecular level, then an interaction observed with the more commonly used drug in the pair can be inferred as a potential interaction for its less-frequently used counterpart.  


---
## 📄 Report

- [Read the full report (PDF)](docs/HIS%20Final%20report.pdf)

---

## Video

- [Watch video demonstration here (mov)](video/project_demo.mov)

---
-
## 📑 Project Overview

Drug interactions remain a major cause of **adverse drug events (ADEs)**, leading to avoidable patient harm and increased healthcare costs.  
Our project proposes a **machine learning-powered Clinical Decision Support System (DSS)** that can be integrated into **Computerized Physician Order Entry (CPOE)** systems to proactively identify potential drug interactions.

**Key Contributions:**
- Developed a DSS trained on the **Higher-Order Drug-Drug Interaction Dataset** (FDA Adverse Event Reporting System).  
- Designed a **database system** with cleaned, structured drug data (pharmacodynamics, pharmacokinetics, regulatory data, timestamps).  
- Proposed integration of **Retrieval-Augmented Generation (RAG)** to continuously incorporate the latest literature into interaction detection.  
- Aligned system design with **RxNorm**, **HL7**, **HIPAA**, and other healthcare IT standards.  
- Conducted a **policy and stakeholder analysis** to evaluate challenges and implications for implementation.  

---

## 🏗️ System Architecture

- **Drug Database:** Structured into multiple tables (drugs, drug_indications, properties, pharmacodynamics, timestamps, regulatory info).  
- **DSS Model:** Predicts interaction risk levels (low, moderate, high) based on drug features.  
- **User Interface:** Provides intuitive risk flags and potential drug alternatives directly within prescribing workflows.  

---

## ⚖️ Policy & Deployment Considerations

- Addresses issues of **alert fatigue** by implementing context-aware filtering and tiered alerts.  
- Ensures compliance with **HIPAA privacy/security rules** and **HL7 interoperability standards**.  
- Recommended **incremental pilot deployment** (e.g., cardiology, oncology) before full-scale rollout.  
- Governance-level steering committee proposed for monitoring, retraining, and feedback.  

---

## 👥 Authors

- **David Forson**  
- **Mohammed Moidul Hassan**  
- **Chiara Zheng**

*Carnegie Mellon University, Heinz College — Spring 2025*

---

## 📌 How to Use This Repo
```
- Navigate to the `docs/` folder to access the full report.  
- Explore the PDF for detailed methodology, architecture diagrams, and policy analysis.  
- Future versions may include the implementation code, datasets, and Streamlit demo of the DSS interface.
```

```
- Navigate to the `video/` folder to access a video demonstration.
```
---

## 📚 References

Selected references from the report:
- Beninger, P. (2023). *Drug-Drug Interactions: How to Manage the Risk–A Stakeholder Approach*. Clinical Therapeutics.  
- Hammar, T., et al. (2021). *Providing Drug–Drug Interaction Services for Patients—A Scoping Review*. Pharmacy Journal.  
- CMS.gov — *Promoting Interoperability Programs*.  

(See full reference list in the report PDF.)

---
