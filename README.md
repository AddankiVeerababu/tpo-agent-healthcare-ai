# TPO Agent: Governed Agentic AI for Healthcare Decision Support

TPO Agent is a proof-of-concept healthcare AI application built for a Gen AI Researcher Intern assessment. The project demonstrates how Agentic AI, retrieval, machine learning, pattern recognition, anomaly detection, local LLM reasoning, and governance controls can support synthetic healthcare Treatment, Payment, and Operations workflows.

The app accepts a synthetic healthcare case, retrieves relevant fictional policy evidence, predicts risk, identifies claim patterns, checks anomaly signals, generates a policy-aware explanation, applies human-in-the-loop governance rules, and produces a structured reviewer recommendation.

This project uses synthetic data only. It does not use real patient data, Protected Health Information, real medical records, real claims, or real payer policy documents. It is for educational and assessment purposes only and must not be used for real clinical, billing, coverage, payment, or operational decisions.

---

## Assessment Topic

Clinical Decision Making and Pattern Recognition in Health Care: Chain Reasoning, Agentic Generative AI, Classification, Prediction, Inference, Clustering, and Time-Series Anomaly Detection for Treatment, Payment, and Operations.

---

## Project Title

TPO Agent: Governed Agentic AI for Healthcare Decision Support

---

## What TPO Means

TPO stands for Treatment, Payment, and Operations.

In this project:

* Treatment means the clinical context of the case, such as diagnosis, procedure, provider type, and medical necessity documentation.
* Payment means claim review, payment integrity, documentation completeness, claim amount, and review recommendation.
* Operations means reviewer triage, case routing, pattern detection, anomaly monitoring, and workflow prioritization.

---

## Main Problem

Healthcare case review is not just a prediction task. A reviewer needs to understand:

* What happened in the case
* What policy evidence is relevant
* Whether documentation is complete
* Whether the case looks risky or unusual
* Why the model produced a risk signal
* Whether the case should be approved, manually reviewed, or routed for documentation
* Whether a human reviewer is required

A simple chatbot may not be grounded in evidence. A simple machine learning model may not explain itself. TPO Agent combines both approaches inside a governed workflow.

---

## What the Prototype Does

The user enters a synthetic healthcare case, such as:

```text
Patient age: 67
Diagnosis: Type 2 diabetes with neuropathy
Procedure: Wound care
Claim amount: 1250
Prior visits in last 30 days: 6
Documentation complete: No
Provider type: Outpatient clinic
Workflow type: Payment
```

The system then runs a multi-step agentic workflow:

```text
Case Intake
→ Policy Retrieval
→ Risk Classification
→ Claim Pattern Clustering
→ Claim-Level Anomaly Detection
→ Time-Series Operations Signal
→ Explainability
→ Local LLM Reasoning
→ Governance Review
→ Final Reviewer Dashboard
```

The final output includes:

* Decision recommendation
* Risk level
* Claim anomaly signal
* Claim pattern cluster
* Time-series operations signal
* Retrieved policy evidence
* Explainability summary
* Local LLM reasoning
* Governance flags
* Reviewer Q&A assistant
* Dataset monitoring charts

Final decision labels:

```text
Approve
Manual Review
Request Documentation
```

The system does not autonomously deny care, deny payment, or make final clinical or payment decisions.

---

## Final App Layout

The Streamlit app is organized into separate tabs to make it feel like a real reviewer workflow:

```text
1. Case Intake
2. Decision
3. Evidence
4. Pattern Signals
5. Reviewer Assistant
6. Data Monitor
```

### Case Intake

The user enters the synthetic healthcare case.

### Decision

Shows the final recommendation, risk level, anomaly status, human review requirement, recommended next step, governance flags, and technical trace.

### Evidence

Shows retrieved fictional policy documents and the LLM-generated case reasoning.

### Pattern Signals

Shows clustering, claim-level anomaly detection, time-series operations signal, explainability summary, and model signals.

### Reviewer Assistant

Allows the reviewer to ask case-specific questions such as:

```text
Why was this decision recommended?
What policy evidence supports the decision?
What cluster does this case belong to?
Is there a time-series anomaly for this procedure?
What should the human reviewer check next?
Is this an autonomous denial?
```

### Data Monitor

Shows synthetic dataset charts and synthetic time-series claim activity.

---

## Agentic Workflow

The backend is organized into separate agents:

| Agent                     | Purpose                                                                     |
| ------------------------- | --------------------------------------------------------------------------- |
| Case Intake Agent         | Validates the incoming synthetic case using Pydantic                        |
| Policy Retrieval Agent    | Retrieves fictional policy evidence using ChromaDB and SentenceTransformers |
| Risk Classification Agent | Predicts Low, Medium, or High risk using Scikit-learn                       |
| Clustering Agent          | Groups the case into a synthetic claim pattern cluster using KMeans         |
| Claim Anomaly Agent       | Detects unusual claim-level patterns using Isolation Forest                 |
| Time-Series Anomaly Agent | Checks synthetic daily claim activity for operational spikes                |
| Explainability Agent      | Produces a practical SHAP-style feature importance summary                  |
| LLM Reasoning Agent       | Uses Ollama to generate policy-aware reasoning                              |
| Governance Agent          | Adds human review, safety, and responsible AI flags                         |

LangGraph is used to orchestrate the workflow.

---

## Technologies Used

| Technology           | Purpose                                           |
| -------------------- | ------------------------------------------------- |
| Python               | Main programming language                         |
| Streamlit            | Interactive dashboard                             |
| Pandas               | Data loading and preprocessing                    |
| Scikit-learn         | Classification, clustering, and anomaly detection |
| Random Forest        | Risk classification                               |
| KMeans               | Claim pattern clustering                          |
| Isolation Forest     | Claim-level and time-series anomaly detection     |
| ChromaDB             | Local vector database for policy retrieval        |
| SentenceTransformers | Embeddings for semantic search                    |
| Ollama               | Local LLM reasoning and reviewer Q&A              |
| Plotly               | Dashboard charts                                  |
| Pydantic             | Structured input and output validation            |
| LangGraph            | Agent workflow orchestration                      |
| SHAP                 | Explainability dependency and future extension    |
| Pytest               | Automated testing                                 |
| Docker               | Containerization support                          |

---

## Gen AI and ML Concepts Demonstrated

This project demonstrates:

* Agentic AI workflow
* Chain reasoning
* Retrieval-Augmented Generation
* Local LLM reasoning
* Classification
* Prediction
* Inference
* KMeans clustering
* Claim-level anomaly detection
* Time-series anomaly detection
* Explainability
* Human-in-the-loop review
* Governance controls
* Reviewer Q&A
* Audit-style technical trace

---

## Repository Structure

```text
tpo-agent-healthcare-ai/
│
├── README.md
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── submission_checklist.md
│
├── app/
│   ├── __init__.py
│   ├── streamlit_app.py
│   ├── config.py
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── graph.py
│   │   ├── case_intake_agent.py
│   │   ├── policy_retrieval_agent.py
│   │   ├── risk_classification_agent.py
│   │   ├── clustering_agent.py
│   │   ├── anomaly_detection_agent.py
│   │   ├── timeseries_anomaly_agent.py
│   │   ├── shap_explanation_agent.py
│   │   ├── llm_reasoning_agent.py
│   │   └── governance_agent.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_models.py
│   │   ├── train_advanced_models.py
│   │   ├── classifier.pkl
│   │   ├── anomaly_model.pkl
│   │   ├── preprocessing.pkl
│   │   ├── clustering_model.pkl
│   │   ├── timeseries_anomaly_model.pkl
│   │   └── timeseries_preprocessing.pkl
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── build_vector_db.py
│   │   └── retriever.py
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── decision_schema.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── visualization.py
│   │
│   └── data/
│       ├── synthetic_claims.csv
│       ├── synthetic_claims_timeseries.csv
│       └── policy_docs/
│           ├── wound_care_policy.txt
│           ├── imaging_policy.txt
│           ├── documentation_policy.txt
│           ├── payment_integrity_policy.txt
│           └── human_review_policy.txt
│
├── chroma_db/
│   └── .gitkeep
│
├── tests/
│   ├── test_classifier.py
│   ├── test_anomaly.py
│   ├── test_retriever.py
│   └── test_schema.py
│
├── report/
│   └── TPO_Agent_Cotiviti_Final_Human_Report.docx
│
├── presentation/
│   └── TPO_Agent_Cotiviti_Aligned_Final_Presentation.pptx
│
└── video/
    └── demo_video_script.md
```

---

## Synthetic Data

The project uses two synthetic datasets.

### Claim-Level Dataset

```text
app/data/synthetic_claims.csv
```

This contains synthetic claim-level rows with:

* Case ID
* Patient age
* Diagnosis
* Procedure
* Claim amount
* Prior visits in last 30 days
* Documentation completeness
* Provider type
* Length of stay
* Synthetic member risk score
* Workflow type
* Risk level

### Time-Series Dataset

```text
app/data/synthetic_claims_timeseries.csv
```

This contains synthetic daily operational claim activity with:

* Date
* Procedure
* Claim count
* Total claim amount
* Incomplete documentation count
* Average member risk score

The time-series file is used to demonstrate lightweight operational anomaly detection.

---

## Synthetic Policy Documents

The RAG layer retrieves from fictional policy documents located in:

```text
app/data/policy_docs/
```

The policy documents include:

* Wound care policy
* Imaging policy
* Documentation completeness policy
* Payment integrity policy
* Human review and governance policy

These documents are fictional and used only for this synthetic demo.

---

## How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/tpo-agent-healthcare-ai.git
cd tpo-agent-healthcare-ai
```

Replace `YOUR-GITHUB-USERNAME` with your GitHub username.

---

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Train the Base ML Models

```bash
python app/models/train_models.py
```

This creates:

```text
app/models/classifier.pkl
app/models/anomaly_model.pkl
app/models/preprocessing.pkl
```

---

### 5. Train the Advanced Pattern Models

```bash
python app/models/train_advanced_models.py
```

This creates:

```text
app/models/clustering_model.pkl
app/models/timeseries_anomaly_model.pkl
app/models/timeseries_preprocessing.pkl
```

---

### 6. Build the Vector Database

```bash
python app/rag/build_vector_db.py
```

This creates a local ChromaDB collection from the fictional policy documents.

---

### 7. Run the Streamlit App

```bash
python -m streamlit run app/streamlit_app.py --server.port 8511
```

Open:

```text
http://localhost:8511
```

---

## Ollama Setup

The app uses Ollama for local LLM reasoning and the Reviewer Assistant.

Install Ollama and pull the model:

```bash
ollama pull llama3.2:3b
```

Check available models:

```bash
ollama list
```

If Ollama is unavailable, the app uses a safe fallback response instead of crashing.

---

## How to Use the App

1. Open the app in the browser.
2. Go to the Case Intake tab.
3. Use the default wound care case or enter a new synthetic case.
4. Click Run Case Review.
5. Open the Decision tab to review the final recommendation.
6. Open the Evidence tab to inspect retrieved policy evidence.
7. Open the Pattern Signals tab to view clustering, claim anomaly, time-series anomaly, and explainability.
8. Open the Reviewer Assistant tab and ask case-specific questions.
9. Open the Data Monitor tab to view synthetic data charts.

Recommended demo questions:

```text
Why was this decision recommended?
What policy evidence supports the decision?
What cluster does this case belong to?
Is there a time-series anomaly for this procedure?
What should the human reviewer check next?
Is this an autonomous denial?
```

---

## Example Output

For the default wound care case, the app may return:

```text
Decision: Request Documentation
Risk Level: High
Claim Anomaly: No
Human Review: Yes
Claim Pattern Cluster: Moderate treatment/utilization claim pattern
Time-Series Signal: No recent operational spike detected
Recommended Action: Request additional documentation before final payment or operational decision
```

The final result may vary slightly depending on local model behavior, but the workflow remains the same.

---

## Testing

Run the automated tests:

```bash
pytest
```

The test suite validates:

* Classifier model loading and prediction
* Isolation Forest anomaly output
* RAG retriever behavior
* Pydantic schema validation

Recommended manual checks:

```text
1. Train base models
2. Train advanced models
3. Build vector DB
4. Run Streamlit app
5. Run default case
6. Open all tabs
7. Ask a Reviewer Assistant question
8. Confirm policy evidence and governance flags appear
```

---

## Docker Usage

Build the Docker image:

```bash
docker build -t tpo-agent-healthcare-ai .
```

Run the container:

```bash
docker run -p 8501:8501 tpo-agent-healthcare-ai
```

Open:

```text
http://localhost:8501
```

Note: Ollama usually runs outside the Docker container unless configured as a separate service. If Ollama is unavailable inside Docker, the app uses fallback reasoning.

---

## Responsible AI and Governance

This prototype includes explicit safeguards:

* Synthetic data only
* No PHI used
* Fictional policies only
* Recommendation only
* No autonomous denial
* Human review required for high-risk or incomplete documentation cases
* Policy evidence required
* Explainability included
* Technical trace available
* Final decision remains with a qualified human reviewer

---

## What the Prototype Proves

The prototype shows that a healthcare AI workflow can be designed as a reviewer-assist system rather than an autonomous decision-maker.

It demonstrates that:

* RAG can ground reasoning in policy evidence.
* ML can identify risk and pattern signals.
* Clustering can group similar synthetic claim patterns.
* Anomaly detection can flag unusual claim-level or operational activity.
* Local LLMs can explain case outputs in plain language.
* Governance rules can keep the final recommendation controlled and reviewable.

---

## Limitations

This is a hackathon-style proof of concept, not a production healthcare AI system.

Current limitations:

* Small synthetic dataset
* Fictional policy documents
* No real claims data
* No PHI
* No EHR or FHIR integration
* No production authentication
* No role-based access control
* No persistent audit database
* Simplified SHAP-style explanation
* Lightweight time-series anomaly detection
* No clinical or financial validation
* Local LLM behavior depends on Ollama availability

---

## Future Improvements

Potential next steps:

* Larger synthetic or de-identified datasets
* Full SHAP TreeExplainer visualizations
* Real audit logging
* Role-based reviewer dashboard
* Reviewer feedback loop
* Model monitoring and drift detection
* Stronger LLM output validation
* FHIR-style synthetic clinical records
* More realistic time-series forecasting
* Integration with case queue management
* Separate reviewer, manager, and auditor views

---

## Deliverables

The final submission includes:

```text
1. Written report
2. Hackathon proof-of-concept code
3. PowerPoint presentation
4. Recorded MP4 video demo
```

The GitHub repository should include:

```text
/report
/presentation
/video
/app
/tests
README.md
requirements.txt
Dockerfile
submission_checklist.md
```

---

## Disclaimer

This project is a synthetic educational proof of concept. It must not be used for real medical, clinical, billing, insurance, coverage, payment, or operational decisions.

No real patient data, no PHI, no real claims data, and no real healthcare policy data are used.
