# SLM-QA-Model
# Small Language Model (SLM) for Question Answering

## Overview
This project implements a **Small Language Model (SLM)** designed to answer questions based on the content of a given book. The model employs **DistilBERT** for efficient natural language understanding and **FAISS** for rapid text retrieval. It processes book text into manageable chunks, retrieves relevant context for a query, and generates responses accurately.

## Approach
The implementation follows a **retrieval-augmented generation (RAG) style approach**, consisting of:

1. **Text Preprocessing**: Chunking the input book text into manageable overlapping segments for efficient retrieval.
2. **Indexing with FAISS**: Creating vector embeddings for the chunks and storing them in a FAISS index for fast retrieval.
3. **Question Answering with DistilBERT**: Retrieving relevant chunks based on query embeddings and generating answers using DistilBERT.
4. **Evaluation Metrics**: Assessing performance using **Exact Match (EM)** and **F1-score**.

## Model Architecture
The model consists of the following components:

### 1. **Text Preprocessing**
- The input book text is split into chunks of **512 words**, with an overlap of **50 words** between consecutive chunks to maintain context.
- The text is tokenized using **Hugging Face's `AutoTokenizer`**.

### 2. **Vector Indexing with FAISS**
- FAISS is used for **fast nearest-neighbor search** on text embeddings.
- Each chunk is converted into a **768-dimensional embedding** (using the output from the DistilBERT transformer model) and indexed.

### 3. **Question Answering Using DistilBERT**
- The model searches the FAISS index for the most relevant chunk corresponding to the input question.
- The top **retrieved chunk** is passed into DistilBERT along with the question.
- The answer is extracted based on **start and end logits** predicted by the model.

### 4. **Evaluation Metrics**
- The model is evaluated on a **SQuAD-style dataset** using:
  - **Exact Match (EM):** Measures how often the predicted answer exactly matches the ground truth.
  - **F1 Score:** Measures the overlap between the predicted and ground-truth answer, considering both precision and recall.

---

## Installation and Dependencies
Ensure you have the required dependencies installed:

```bash
pip install torch transformers faiss-cpu evaluate
```

---

## Instructions for Running the Model

### **1. Initialize the Model**
```python
slm = SLMQuestionAnswering()
```

### **2. Provide Book Text as Context**
```python
book_text = "This is a sample book content for testing our SLM-based QA model. The model should accurately retrieve and answer questions from the given text."
slm.build_index(book_text)
```

### **3. Ask a Question**
```python
question = "What should the model do?"
answer = slm.answer_question(question)
print("Answer:", answer)
```

### **4. Evaluate Model Performance**
```python
from evaluate import load

dataset = [
    {
        "id": "1",
        "question": "Where does the sun rise?",
        "answers": {"text": ["in the east"], "answer_start": [13]},
    },
    {
        "id": "2",
        "question": "What does the sun provide?",
        "answers": {"text": ["light and energy"], "answer_start": [38]},
    }
]

results = slm.evaluate(dataset)
print("Exact Match (EM):", results["exact_match"])
print("F1 Score:", results["f1"])
```

---

## Sample Outputs
### **Input Question:**
```text
What should the model do?
```
### **Generated Answer:**
```text
Accurately retrieve and answer questions from the given text.
```
### **Evaluation Output:**
```text
Exact Match (EM): 50.0
F1 Score: 83.5
```

---

## GitHub Repository
The full implementation and dataset are available at:
[GitHub Repository](https://github.com/your-repo-link)

---

## Observations & Key Learnings
1. **DistilBERT performs efficiently** in a low-compute setting while maintaining reasonable accuracy.
2. **FAISS indexing significantly speeds up retrieval**, making the model scalable for large book-sized contexts.
3. **Text chunking with overlap is crucial** for maintaining context while retrieving answers.
4. **Evaluation metrics provide meaningful insights** into the model's accuracy and recall performance.
5. **Potential improvements:**
   - Implement **multi-turn conversation support**.
   - Extend to **larger context sizes** with **RAG-based approaches**.
   - Deploy the model using **FastAPI or Flask** for real-world applications.

---

## Future Enhancements
- **Long-context handling:** Improve retrieval efficiency for large documents.
- **RAG Integration:** Use a transformer-based retriever (e.g., `sentence-transformers`).
- **Web Deployment:** Provide an API interface for easy querying.

---

## Conclusion
This SLM-based QA system effectively retrieves and generates accurate answers from book-sized texts. By leveraging **FAISS indexing** and **DistilBERT**, it achieves **fast retrieval** and **precise response generation**, making it a practical solution for real-world applications like **chatbots, document search, and knowledge extraction**.

---
**Author:** Saumya Singh Jaiswal  
**Date:** February 2025

