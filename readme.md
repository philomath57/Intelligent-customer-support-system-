# 🎧 Customer Support Intelligence & Ticket Automation System

An end-to-end **AI-powered customer support system** that combines machine learning, transformer-based NLP, semantic search, cross-encoder reranking, and Retrieval-Augmented Generation (RAG) to analyse customer support tickets and generate relevant support responses.

The system predicts the **support queue and priority** of a customer query, retrieves semantically similar historical support tickets, reranks the most relevant cases, and uses the retrieved information to generate a suggested customer support response.

---

## 📌 Project Overview

Customer support teams receive large numbers of tickets covering technical issues, billing problems, product queries, service outages, returns, sales enquiries, and other customer concerns.

Manually analysing and routing every ticket can be time-consuming and inconsistent.

This project develops an intelligent support pipeline that can:

* Classify customer queries into the appropriate support queue.
* Predict the priority level of a customer query.
* Find historically similar support tickets.
* Rerank retrieved tickets according to their relevance.
* Use previous support answers as contextual knowledge.
* Generate an AI-assisted customer support response.
* Provide the results through an interactive Streamlit application.

The system is designed as a **decision-support and response-assistance tool**, with generated responses intended for human review rather than fully autonomous customer communication.

---

# 🚀 Key Features

### 1. Customer Query Analysis

The application accepts a customer issue through a simple Streamlit interface.

Example:

> "I was charged twice for my monthly subscription. Both charges appear on my account, but I only made one purchase."

The system processes the query through multiple NLP components.

---

### 2. Support Queue Classification

A fine-tuned **BERT-based text classification model** predicts the appropriate customer-support queue.

The project contains the following queue categories:

| Label | Support Queue                   |
| ----: | ------------------------------- |
|     0 | Technical Support               |
|     1 | Product Support                 |
|     2 | Customer Service                |
|     3 | IT Support                      |
|     4 | Billing and Payments            |
|     5 | Returns and Exchanges           |
|     6 | Service Outages and Maintenance |
|     7 | Sales and Pre-Sales             |
|     8 | Human Resources                 |
|     9 | General Inquiry                 |

The queue classifier uses:

**Model:** `bert-base-uncased`

**Task:** Multi-class text classification

**Output:** Predicted support queue

---

### 3. Priority Classification

A machine-learning pipeline predicts the priority of the customer query.

The available priority levels are:

* Low
* Medium
* High

The project experiments with several machine-learning approaches using TF-IDF features, including:

* Logistic Regression
* Linear Support Vector Classification
* Random Forest
* XGBoost

The selected Random Forest + TF-IDF pipeline is saved and used by the Streamlit application.

---

### 4. Semantic Ticket Retrieval

The system uses **Sentence-BERT** to identify historical support tickets that are semantically similar to the customer's query.

The implementation uses:

**Model:** `all-MiniLM-L6-v2`

The ticket text is converted into dense vector representations.

These embeddings are stored in a **FAISS** similarity index.

The system initially retrieves the top 10 semantically similar historical tickets.

---

### 5. Cross-Encoder Reranking

Semantic similarity provides candidate tickets, but the most similar embedding is not necessarily the most useful support example.

Therefore, the retrieved candidates are passed through a Cross-Encoder.

**Model:**

`cross-encoder/ms-marco-MiniLM-L6-v2`

The Cross-Encoder evaluates the relationship between:

```text
Customer Query + Historical Ticket
```

The candidates are then sorted according to their reranker score, and the top 3 results are used for the RAG stage.

---

### 6. Retrieval-Augmented Generation

The system uses a RAG pipeline to provide historical support information to the language model.

The RAG pipeline consists of:

```text
Customer Query
       ↓
Sentence-BERT Retrieval
       ↓
Top 10 Similar Tickets
       ↓
Cross-Encoder Reranking
       ↓
Top 3 Relevant Tickets
       ↓
RAG Context
       ↓
FLAN-T5
       ↓
Suggested Support Response
```

The RAG context contains information such as:

* Subject
* Customer issue
* Support queue
* Priority
* Ticket type
* Previous support answer

This allows the generator to use previous support cases as contextual information instead of generating a response without reference to the historical dataset.

---

### 7. AI-Generated Support Response

The generation component uses:

**Model:** `google/flan-t5-base`

The model receives:

* The customer's query
* The retrieved historical support cases
* Instructions for producing a professional response

The prompt instructs the model to:

* Address the customer's issue directly.
* Use previous support cases as guidance.
* Avoid inventing unsupported information.
* Avoid claiming that actions have already been completed.
* Request human support review when the retrieved information is insufficient.
* Produce a concise and professional response.

---

# 🏗️ System Architecture

```text
                    Customer Query
                          │
                          ▼
              ┌───────────────────────┐
              │   Streamlit Interface  │
              └───────────┬───────────┘
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
     Priority Prediction        Queue Classification
             │                         │
             ▼                         ▼
       TF-IDF + RF                  BERT
             │                         │
             └────────────┬────────────┘
                          │
                          ▼
                 Semantic Retrieval
                          │
                          ▼
                Sentence-BERT Embedding
                          │
                          ▼
                    FAISS Index
                          │
                          ▼
                  Top 10 Candidates
                          │
                          ▼
                 Cross-Encoder
                    Reranking
                          │
                          ▼
                   Top 3 Cases
                          │
                          ▼
                  RAG Context
                          │
                          ▼
                    FLAN-T5
                          │
                          ▼
             Suggested Support Response
```

---

# 📊 Dataset

The project uses the **Multilingual Customer Support Tickets** dataset.

The English-language records are selected for the main classification and retrieval pipeline.

The dataset contains information including:

* `subject`
* `body`
* `answer`
* `type`
* `queue`
* `priority`
* `language`
* `version`
* `tag_1` – `tag_8`
* `body_preprocessed`
* `priority_label`
* `body_vector`

Only English-language tickets are used in the implemented pipeline.

---

# 🧠 Machine Learning Pipeline

## Queue Classification

The queue-classification stage experimented with:

```text
Customer Ticket
      ↓
Text Tokenisation
      ↓
BERT
      ↓
Classification Head
      ↓
10 Support Queues
```

The BERT model is saved locally and loaded during Streamlit inference.

---

## Priority Classification

The priority pipeline follows:

```text
Customer Ticket
      ↓
TF-IDF Vectorisation
      ↓
Random Forest Classifier
      ↓
Priority Level
```

The trained pipeline is stored as:

```text
priority_classification_model/
└── rf_tfidf_pipeline_priority.pkl
```

---

# 🔎 Semantic Retrieval

The semantic retrieval component converts customer tickets into vector representations using Sentence-BERT.

The embeddings are indexed using FAISS.

```text
Historical Support Tickets
            ↓
Sentence-BERT
            ↓
Dense Embeddings
            ↓
FAISS Index
```

When a new customer query is entered:

```text
New Customer Query
        ↓
Sentence-BERT
        ↓
Query Embedding
        ↓
FAISS Similarity Search
        ↓
Top 10 Historical Tickets
```

The retrieval stage uses vector similarity to identify tickets with similar semantic meaning.

---

# 🔄 Reranking

The top retrieved candidates are subsequently passed to a Cross-Encoder.

```text
Query
  +
Historical Ticket
       ↓
Cross-Encoder
       ↓
Relevance Score
```

The candidates are sorted according to their reranker score.

The top 3 candidates are then passed to the RAG component.

---

# 🤖 RAG Generation

The RAG component combines the user's query with retrieved historical support information.

Example structure:

```text
Customer Issue:
I was charged twice for my monthly subscription.

Previous Support Cases:

Subject: Problem with Monthly Subscription Overcharging

Customer Issue:
...

Queue:
Billing and Payments

Priority:
High

Previous Support Answer:
...
```

This information is provided to FLAN-T5 as contextual knowledge.

The objective is to produce a response grounded in previously available support information.

---

# 📈 Retrieval Evaluation

The semantic retrieval component was evaluated using:

### Top-1 Accuracy

Measures how often the highest-ranked retrieved ticket belongs to the correct support category.

### Top-5 Hit Rate

Measures whether at least one relevant ticket appears within the top five retrieved results.

### Precision@5

Measures the proportion of the five retrieved tickets that belong to the relevant category.

### Recall@5

Measures whether a relevant category is retrieved within the top five results according to the project's evaluation definition.

The implemented retrieval evaluation produced the following results:

| Metric         | Result |
| -------------- | -----: |
| Top-1 Accuracy | 0.7763 |
| Top-5 Hit Rate | 0.8883 |
| Precision@5    | 0.4526 |
| Recall@5       | 0.8883 |

These results indicate that the semantic retrieval stage can identify relevant historical support cases for a substantial proportion of customer queries.

---

# 🖥️ Streamlit Application

The project provides an interactive Streamlit interface.

The user enters a customer query:

```text
Please enter your query
```

The application then displays:

### Customer Query Priority Level

Example:

```text
High
```

### Customer Query Type

Example:

```text
Billing and Payments
```

### Retrieved Results

The application displays the retrieved historical support tickets together with their similarity and ranking information.

### LLM Response

The application generates an AI-assisted customer support response using the RAG pipeline.

---

# 📁 Project Structure

The project can be organised as follows:

```text
customer-support-intelligence/
│
├── app.py
├── requirements.txt
├── README.md
│
├── priority_prediction_helper.py
├── semantic_retrieval.py
├── rag.py
│
├── priority_classification_model/
│   └── rf_tfidf_pipeline_priority.pkl
│
├── bert_model_for_queue_classification/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── ...
│
├── semantic_search_using_BERT/
│   ├── support_ticket_faiss.index
│   ├── retrieval_metadata_df.pkle
│   └── semantic_bert_model/
│       └── ...
│
└── data/
    └── aa_dataset-tickets-multi-lang-5-2-50-version.csv
```

The exact generated model files may vary depending on the model-saving configuration.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

```bash
cd customer-support-intelligence
```

---

## 2. Create a Virtual Environment

Using Python:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

The main dependencies include:

```text
streamlit
pandas
numpy
scikit-learn
joblib
transformers
torch
sentence-transformers
faiss-cpu
xgboost
spacy
huggingface-hub
```

---

# ▶️ Running the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

Streamlit will launch the application in the browser.

Enter a customer support query and select:

```text
Predict and Retrieve Data
```

The application will then execute the complete pipeline.

---

# 🧪 Example Queries

## Example 1 — Technical Support

```text
I am unable to log in to my account. The login page keeps showing an error message even though I am entering the correct password. I have tried resetting my password, but I still cannot access my account.
```

The system can use this query to test:

* Queue classification
* Priority classification
* Semantic retrieval
* Cross-Encoder reranking
* RAG response generation

---

## Example 2 — Billing

```text
I was charged twice for my monthly subscription this month. Both transactions appear on my bank statement, but I only have one active subscription. Could you please check the duplicate charge and let me know how it can be resolved?
```

This query can be used to test the system's ability to retrieve historically similar billing cases and generate a contextual response.

---

# 🛠️ Technologies Used

| Technology                | Purpose                                |
| ------------------------- | -------------------------------------- |
| Python                    | Main programming language              |
| Pandas                    | Data processing                        |
| NumPy                     | Numerical operations                   |
| Scikit-learn              | Traditional machine learning           |
| XGBoost                   | Machine-learning experimentation       |
| PyTorch                   | Deep learning backend                  |
| Hugging Face Transformers | BERT and FLAN-T5                       |
| Sentence Transformers     | Semantic embeddings and Cross-Encoder  |
| FAISS                     | Vector similarity search               |
| Joblib                    | Model persistence                      |
| Streamlit                 | Web application                        |
| Git/GitHub                | Version control and project management |

---

# 🧩 Models Used

| Component               | Model / Method                        |
| ----------------------- | ------------------------------------- |
| Queue Classification    | BERT (`bert-base-uncased`)            |
| Priority Classification | TF-IDF + Random Forest                |
| Semantic Retrieval      | `all-MiniLM-L6-v2`                    |
| Vector Search           | FAISS                                 |
| Reranking               | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Response Generation     | `google/flan-t5-base`                 |

---

# 🔐 Responsible AI Considerations

The generated response is intended as **AI-assisted support**, rather than an autonomous decision-making system.

The RAG prompt specifically instructs the generator not to:

* Invent policies.
* Invent prices.
* Invent refund procedures.
* Claim that an action has been completed without supporting evidence.
* Provide unsupported technical information.

When the retrieved information is insufficient, the system is instructed to recommend review by a human support agent.

This human-in-the-loop approach is particularly important when customer support responses could result in financial, account, technical, or service-related consequences.

---

# ⚠️ Limitations

The current implementation has several limitations.

### Dataset limitations

The system depends on the quality and coverage of historical support tickets. If a new issue is substantially different from the available historical cases, retrieval quality may decrease.

### Classification limitations

The queue and priority models can make incorrect predictions, particularly for ambiguous or unusual customer queries.

### Retrieval limitations

Semantic similarity does not guarantee that a retrieved ticket contains the correct solution.

### Generation limitations

FLAN-T5-base is a relatively lightweight generative model. Its responses may sometimes be generic, incomplete, or overly influenced by the retrieved context.

### Human Review

Generated responses should therefore be reviewed by a support agent before being sent to a customer in a production environment.

---

# 🔮 Future Improvements

Several improvements could extend the system.

### 1. Improved Language Model

Replace FLAN-T5-base with a stronger instruction-following language model capable of better contextual reasoning and response generation.

### 2. Improved RAG Pipeline

Future versions could incorporate:

* Better document chunking
* Metadata filtering
* Hybrid keyword + semantic search
* Improved reranking
* Query rewriting
* Context compression

### 3. Confidence Scores

The application could display confidence or relevance indicators for:

* Queue prediction
* Priority prediction
* Semantic retrieval
* Reranking

### 4. Human Feedback

Support agents could approve, edit, or reject generated responses.

These interactions could subsequently be used to evaluate and improve the response-generation pipeline.

### 5. Multilingual Support

The source dataset contains multilingual customer support tickets. Future versions could extend the current English-focused implementation to multiple languages.

### 6. Conversation History

The application could be extended from single-ticket processing to multi-turn customer conversations.

### 7. Production Deployment

The application could be deployed using a cloud platform and connected to a production database or customer-support platform.

---

# 📌 Project Workflow

The complete implemented workflow is:

```text
                ┌─────────────────────┐
                │   Customer Query    │
                └──────────┬──────────┘
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
      Priority Model              BERT Classifier
             │                           │
             ▼                           ▼
       Priority Level               Queue Type
             │                           │
             └─────────────┬─────────────┘
                           │
                           ▼
                  Sentence-BERT
                           │
                           ▼
                     FAISS Search
                           │
                           ▼
                  Top 10 Candidates
                           │
                           ▼
                   Cross-Encoder
                           │
                           ▼
                    Top 3 Tickets
                           │
                           ▼
                    RAG Context
                           │
                           ▼
                       FLAN-T5
                           │
                           ▼
               Support Response
```

---

# 🎯 Project Objectives

The main objectives of the project are to:

1. Automate customer-support ticket classification.
2. Predict the priority of incoming customer queries.
3. Retrieve relevant historical support cases using semantic similarity.
4. Improve retrieval relevance using cross-encoder reranking.
5. Generate contextual support responses using retrieved information.
6. Provide an interactive AI-assisted customer-support interface.
7. Demonstrate an end-to-end NLP and RAG application using modern machine-learning techniques.

---

# 📚 Learning Outcomes

This project demonstrates practical experience with:

* Natural Language Processing
* Text classification
* Transformer-based models
* BERT fine-tuning
* TF-IDF feature engineering
* Traditional machine learning
* Sentence embeddings
* Vector databases/search
* FAISS
* Semantic similarity
* Cross-Encoder reranking
* Retrieval-Augmented Generation
* Large Language Model prompting
* Streamlit application development
* Model persistence and deployment
* Human-in-the-loop AI systems

---

# 👨‍💻 Author

**Md Faiyaz Alam**

Data Science / Artificial Intelligence

The project demonstrates the design and implementation of an end-to-end NLP-based customer support intelligence system combining traditional machine learning, transformer models, semantic retrieval, reranking, and retrieval-augmented generation.
