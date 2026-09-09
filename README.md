# IntelliAssist AI
IntelliAssist AI is an AI-powered document assistant that allows users to upload documents and ask questions based on their selected files. The system retrieves relevant information from the documents, evaluates the evidence, and generates grounded answers using Generative AI.

## Problem Statement
Users often need to find specific information from large documents such as study notes, reports, reference materials, and business documents. Searching through these documents manually can be time-consuming and may result in incomplete or inaccurate information.

IntelliAssist AI aims to provide a document-based question answering system that can retrieve relevant information from uploaded documents and generate answers based on the available evidence.

## Objective
The main objectives of IntelliAssist AI are:
- Allow users to upload and manage documents.
- Retrieve relevant information from selected documents.
- Improve retrieval accuracy using reranking.
- Generate answers based on retrieved document evidence.
- Avoid generating answers when sufficient evidence is not available.
- Organize documents using workspaces.
- Detect duplicate documents.

## Tools and Technologies Used

### Programming Language
- Python

### Application Framework
- Streamlit

### Document Processing
- PyPDF
- python-docx
- TXT file processing

### Natural Language Processing
- Sentence Transformers
- BGE-M3

### Vector Search
- FAISS

### Reranking
- BAAI/bge-reranker-v2-m3

### Generative AI
- Google Gemini API
- google-genai

### Data Storage
- JSON
- FAISS vector storage

### Deployment and Version Control
- GitHub
- Streamlit Community Cloud

## System Architecture
The system follows a document retrieval and question answering approach:
1. User uploads a PDF, TXT, or DOCX document.
2. The document content is extracted.
3. The extracted content is divided into smaller text chunks.
4. BGE-M3 generates embeddings for the chunks.
5. The embeddings are stored in a FAISS vector index.
6. When the user asks a question, the query is converted into an embedding.
7. FAISS retrieves the most relevant document chunks.
8. The retrieved results are reranked using BAAI/bge-reranker-v2-m3.
9. The retrieved evidence is evaluated using a confidence mechanism.
10. If sufficient evidence is available, Gemini generates the answer.
11. If sufficient evidence is not available, the system returns an insufficient-evidence response.

## Major Functions

### Document Upload
Supports PDF, TXT, and DOCX documents and processes them for question answering.

### Multi-Document Question Answering
Users can select multiple documents and ask questions based on their combined content.

### Semantic Retrieval
BGE-M3 embeddings and FAISS are used to retrieve relevant document content based on semantic similarity.

### Document Reranking
BAAI/bge-reranker-v2-m3 reranks the retrieved results to identify the most relevant evidence.

### Evidence-Based Answer Generation
The system generates answers using the retrieved document content rather than relying only on the model's general knowledge.

### Confidence and Refusal Mechanism
The system evaluates the retrieved evidence before generating an answer. When sufficient evidence is not available, it avoids providing an unsupported answer.

### Duplicate Detection
Uploaded documents are checked using hashing techniques to identify duplicate files or duplicate content.

### Workspace Management
Documents can be organized into different workspaces or folders.

### Profile Isolation
Student and Business profiles use separate document and workspace data so that documents belonging to one profile are not automatically exposed to the other.

## Retrieval Evaluation

The retrieval system was evaluated using a test set containing 10 questions, including answerable and unanswerable questions.

| Retrieval Method | Recall |
|------------------|--------|
| Semantic Retrieval | 87.5% |
| Hybrid Retrieval | 87.5% |
| BGE Reranker | 100% |

The evaluation showed that reranking improved the retrieval results on the tested dataset.

## Output
The system provides:
- Detailed answers to questions
- Short summaries
- Supporting evidence
- Source document information
- Insufficient-evidence responses for unsupported questions

## Live Demo
(https://intelliassist-ai-sukriti.streamlit.app/)

## Project Structure

```text
IntelliAssist-AI/
│
├── app/
│   ├── ingestion/
│   ├── confidence.py
│   ├── access_control.py
│   ├── document_registery.py
│   ├── rag.py
│   ├── sentiment_intent.py
│   └── streamlit_app.py
│
├── data/
│   ├── document_registry.json
│   └── workspace_registry.json
│
├── storage/
│   └── faiss/
│
├── requirements.txt
└── README.md
