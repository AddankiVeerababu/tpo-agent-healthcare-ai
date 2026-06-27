# TPO Agent: Governed Agentic AI for Healthcare Decision Support

TPO Agent is a professional proof-of-concept healthcare informatics application that demonstrates how governed Agentic AI can support synthetic Treatment, Payment, and Operations workflows.

The application accepts a synthetic healthcare claim or clinical/payment case, retrieves relevant synthetic policy evidence using Retrieval-Augmented Generation, predicts risk level using machine learning, detects anomalies, generates explainability output, uses a local LLM for policy-aware reasoning, applies responsible AI governance rules, and produces a structured decision recommendation.

> This project uses synthetic data only. No real patient data or Protected Health Information is used. This application is for educational and assessment purposes only and must not be used for real clinical, billing, coverage, payment, or operational decisions.

---

## Project Title

**TPO Agent: Governed Agentic AI for Healthcare Decision Support**

---

## Project Goal

The goal of this project is to demonstrate an enterprise-style Agentic AI workflow for healthcare decision support across Treatment, Payment, and Operations.

The system shows how an AI workflow can combine:

* Agentic AI orchestration
* Retrieval-Augmented Generation
* Local LLM reasoning
* Risk classification
* Prediction and inference
* Anomaly detection
* Explainable AI
* Human-in-the-loop review
* Responsible AI governance
* Synthetic data and privacy protection

---

## Why This Project Matters

Healthcare organizations often need to review cases for clinical documentation completeness, payment integrity, utilization patterns, medical necessity support, and operational risk. A governed AI system can help triage cases, surface relevant policies, explain model predictions, and route cases to human reviewers when uncertainty or risk is high.

This project does not replace human decision-making. Instead, it demonstrates how AI can support reviewers by combining structured machine learning, policy retrieval, generative reasoning, and governance safeguards.

---

## Agentic Workflow

```text
Case Intake Agent
        ↓
Policy Retrieval Agent
        ↓
Risk Classification Agent
        ↓
Anomaly Detection Agent
        ↓
SHAP Explanation Agent
        ↓
LLM Reasoning Agent
        ↓
Governance Review Agent
        ↓
Final Decision Dashboard
```

---

## What Each Agent Does

### 1. Case Intake Agent

Validates and standardizes the synthetic healthcare case using Pydantic.

### 2. Policy Retrieval Agent

Uses SentenceTransformers and ChromaDB to retrieve relevant synthetic healthcare policy documents.

### 3. Risk Classification Agent

Uses a Scikit-learn Random Forest classifier to predict whether a case is Low, Medium, or High risk.

### 4. Anomaly Detection Agent

Uses Scikit-learn Isolation Forest to identify unusual claim patterns.

### 5. SHAP Explanation Agent

Generates an explainability summary showing the most influential features behind the model prediction.

### 6. LLM Reasoning Agent

Uses Ollama and a local LLM to generate a policy-aware explanation using only the synthetic case facts, retrieved policy evidence, model prediction, anomaly result, and explainability output.

### 7. Governance Review Agent

Applies responsible AI rules, human-in-the-loop flags, safety statements, and final decision logic.

### 8. Final Decision Dashboard

Displays the structured decision, risk level, anomaly signal, policy evidence, reasoning, explainability, governance flags, and synthetic dataset charts in Streamlit.

---

## Technologies Used

| Technology           | Purpose                                                    |
| -------------------- | ---------------------------------------------------------- |
| Python               | Core programming language for the AI workflow              |
| Streamlit            | Interactive dashboard and demo interface                   |
| Pandas               | Synthetic claims data loading and preprocessing            |
| Scikit-learn         | Risk classification and Isolation Forest anomaly detection |
| ChromaDB             | Local vector database for RAG policy retrieval             |
| SentenceTransformers | Embedding model for semantic search                        |
| Ollama               | Local LLM reasoning without external API dependency        |
| Plotly               | Professional dashboard visualizations                      |
| Pydantic             | Structured input and output validation                     |
| SHAP                 | Explainability dependency and future extension             |
| LangGraph            | Agentic workflow orchestration                             |
| Pytest               | Automated test suite                                       |
| Docker               | Reproducible containerized deployment                      |

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
│   │   ├── anomaly_detection_agent.py
│   │   ├── shap_explanation_agent.py
│   │   ├── llm_reasoning_agent.py
│   │   └── governance_agent.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_models.py
│   │   ├── classifier.pkl
│   │   ├── anomaly_model.pkl
│   │   └── preprocessing.pkl
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
│   └── report_outline.md
│
├── presentation/
│   └── presentation_outline.md
│
└── video/
    └── demo_video_script.md
```

---

## Synthetic Data

The project uses a synthetic healthcare claims dataset located at:

```text
app/data/synthetic_claims.csv
```

The dataset includes fields such as:

* Patient age
* Diagnosis
* Procedure
* Claim amount
* Prior visits in last 30 days
* Documentation completeness
* Provider type
* Length of stay
* Synthetic member risk score
* TPO workflow type
* Risk level

No real patient data, claims data, medical records, or PHI are used.

---

## Synthetic Policy Documents

The RAG system retrieves from fictional policy documents located in:

```text
app/data/policy_docs/
```

Policy documents include:

* Wound care review policy
* Imaging review policy
* Documentation completeness policy
* Payment integrity policy
* Human-in-the-loop governance policy

These documents are fictional and created only for this project.

---

## How to Run Locally

### 1. Create and activate virtual environment

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

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Train ML models

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

### 4. Build the vector database

```bash
python app/rag/build_vector_db.py
```

This creates the local ChromaDB vector store used for policy retrieval.

---

### 5. Run Streamlit

```bash
python -m streamlit run app/streamlit_app.py --server.port 8505
```

Open:

```text
http://localhost:8505
```

---

## Ollama Setup

Install Ollama locally and pull a small local model:

```bash
ollama pull llama3.2:3b
```

The app uses Ollama for local LLM reasoning.

If Ollama is unavailable, the application falls back to a safe deterministic explanation so the demo can still run.

---

## Example Case

```text
Patient age: 67
Diagnosis: Type 2 diabetes with neuropathy
Procedure: Wound care
Claim amount: 1250
Prior visits in last 30 days: 6
Documentation complete: No
Provider type: Outpatient clinic
Claim type: Payment
```

---

## Example Output

```text
Decision: Request Documentation
Risk Level: High
Anomaly Detected: No
Policy Evidence: Wound care claims require wound size, wound location, treatment plan, and medical necessity documentation.
Explainability Summary: Missing documentation, claim amount, prior visits, and member risk score influenced the prediction.
Reasoning: The case should be routed for documentation review because documentation is incomplete and the case has elevated risk signals.
Governance Flags: Human review required, recommendation only, no autonomous denial, synthetic data only.
Recommended Action: Request additional documentation before final payment or operational decision.
```

---

## Testing

Run:

```bash
pytest
```

The test suite validates:

* Classifier artifact loading and prediction
* Isolation Forest anomaly model output
* RAG retriever behavior
* Pydantic schema validation

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

Note: Ollama is usually run outside the Docker container unless a separate Ollama service is configured. If Ollama is unavailable inside Docker, the app uses a safe fallback explanation.

---

## Responsible AI and Governance

This project includes responsible AI safeguards:

* Synthetic data only
* No PHI used
* Recommendation only
* No autonomous denial
* Human review required for high-risk cases
* Human review required for incomplete documentation
* Policy evidence required
* Explainability summary included
* Final decision framed as decision support only
* No medical diagnosis generation
* No replacement of licensed clinical judgment

---

## Final Decision Labels

The system can return:

```text
Approve
Manual Review
Request Documentation
```

The application does not autonomously deny claims, deny care, or make final payment decisions.

---

## Assessment Relevance

This project directly demonstrates skills relevant to an Agentic AI Research internship:

* Agentic AI system design
* LangGraph workflow orchestration
* Healthcare informatics reasoning
* Treatment, Payment, and Operations workflow support
* Retrieval-Augmented Generation
* Local LLM integration
* Machine learning classification
* Anomaly detection
* Explainable AI
* Pydantic structured output validation
* Human-in-the-loop governance
* Responsible AI and safety-aware design
* Streamlit dashboard development
* Automated testing with Pytest
* Docker-ready project packaging

---

## Limitations

This is a proof-of-concept and uses a small synthetic dataset. The risk model is not trained on real-world healthcare claims and should not be interpreted as clinically or financially valid.

Future improvements could include:

* Larger synthetic dataset generation
* More realistic temporal utilization patterns
* True time-series anomaly detection
* Full SHAP TreeExplainer visualizations
* More advanced LangGraph branching
* Authentication and audit logging
* Role-based reviewer dashboard
* Model monitoring and drift detection
* FHIR-style synthetic clinical data integration

---

## Disclaimer

This project is a synthetic educational proof-of-concept. It must not be used for real medical, clinical, billing, insurance, coverage, payment, or operational decisions.

No real patient data, no PHI, and no real healthcare policy data are used.
