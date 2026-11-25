# RAG Project: Question Answering over a Cloud Computing and AI Study Guide
 
A Retrieval-Augmented Generation (RAG) system that answers questions from a ~300-page PDF study guide on Cloud Computing and Artificial Intelligence.
 
**Author:** Surabhi Mishra
**Course:** B.Tech CSE (Cloud Computing and Artificial Intelligence), Shri Ramswaroop Memorial University
 
## Overview
 
A language model on its own can give outdated or invented answers. This project grounds its answers in a document: the system finds the most relevant passages in the PDF and gives them to the language model as context, so answers come from the source and can be checked.
 
## Dataset
 
| Item | Details |
| --- | --- |
| File | `cloud_ai_rag_study_guide.pdf` |
| Size | 318 pages |
| Content | 10 chapters, 60 topics, definitions, examples, comparison tables, review questions, answer key, glossary |
| Topics | Cloud basics, containers, storage, security, DevOps, machine learning, neural networks, RAG, MLOps, ethics |
 
## How It Works
 
```
PDF -> Load -> Chunk -> Embed -> Vector Store
                                     |
User question -> Embed -> Retrieve top-k chunks -> LLM + context -> Answer
```
 
1. **Load:** read the PDF and extract text page by page.
2. **Chunk:** split the text into pieces of about 500 to 1000 characters with a small overlap, so ideas are not cut in half.
3. **Embed:** convert each chunk into a vector using an embedding model.
4. **Store:** save the vectors in a vector database (for example FAISS or Chroma).
5. **Retrieve:** embed the user's question and fetch the most similar chunks (top-k, usually 3 to 5).
6. **Generate:** send the question and retrieved chunks to a language model, which writes the final answer.
## Suggested Tech Stack
 
- Python 3.10 or newer
- pypdf or pdfplumber for reading the PDF
- LangChain or LlamaIndex for the pipeline
- FAISS or Chroma as the vector database
- An embedding model and a language model of your choice
## Suggested Project Structure
 
```
rag-project/
├── data/
│   └── cloud_ai_rag_study_guide.pdf
├── src/
│   ├── ingest.py      # load, chunk, embed and store
│   └── query.py       # retrieve and answer
├── requirements.txt
└── README.md
```
 
## Setup
 
```bash
git clone https://github.com/surabhimishra1310/rag-project.git
cd rag-project
pip install -r requirements.txt
```
 
Place the PDF in the `data/` folder, then run:
 
```bash
python src/ingest.py     # builds the vector store (run once)
python src/query.py      # ask questions in the terminal
```
 
Update the commands and file names above to match your actual code.
 
## Example Questions
 
- What is the difference between IaaS and PaaS?
- Why are containers lighter than virtual machines?
- What does ETL stand for?
- How does RAG reduce hallucination?
- Why is chunk overlap useful?
- What is the answer to multiple choice question 4 for Model Deployment? (tests retrieval from the answer key)
## Evaluation Ideas
 
- Use the review questions and the answer key in the PDF as ready-made test cases.
- Check whether the retrieved chunks contain the correct answer (retrieval accuracy).
- Try different chunk sizes (300, 500, 1000) and values of k, and compare results.
- Test questions that are not in the document to confirm the system says it does not know instead of guessing.
## Limitations
 
- Answers depend on retrieval quality. Poor chunking leads to poor answers.
- The study guide repeats some facts in several places (topic pages, tables, glossary), so the same information may be retrieved more than once.
- The document is a practice dataset, not a full textbook.
## Future Improvements
 
- Add source citations with page numbers to every answer
- Build a simple web interface with Streamlit or Gradio
- Try hybrid search (keyword plus vector) and re-ranking
- Support multiple PDFs
## Contact
 
- Email: surabhimishra2005@gmail.com
- GitHub: [surabhimishra1310](https://github.com/surabhimishra1310)
- LinkedIn: [Surabhi Mishra](https://www.linkedin.com/in/surabhi-mishra-29b6a5343)
 
