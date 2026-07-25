# PDF RAG Question Answering System

A Retrieval-Augmented Generation (RAG) system that allows you to ask questions about PDF documents using AI.

## 🚀 Features

- Upload PDF documents
- Ask questions in natural language
- Get AI-powered answers from your documents
- Built with Streamlit, LangChain, and Google Gemini

## 🛠️ Tech Stack

- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: HuggingFace (sentence-transformers)
- **Vector Database**: FAISS
- **Framework**: LangChain
- **UI**: Streamlit
- **Containerization**: Docker

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/PrishaChowdhury/pdf-rag-qa-system.git
cd pdf-rag-qa-system
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Add your GOOGLE_API_KEY to .env
```

5. Run the app:
```bash
streamlit run app.py
```

## 🔑 Getting API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Create an API key
4. Add it to your `.env` file

## 📝 Usage

1. Upload a PDF document
2. Wait for processing
3. Ask questions about the document
4. Get AI-powered answers!

## 🐳 Running with Docker

Build and run the app in a container instead of a local virtual environment:

```bash
docker build -t pdf-rag-qa-system .
docker run -e GOOGLE_API_KEY=your_key -p 8501:8501 pdf-rag-qa-system
```

Then open `http://localhost:8501` in your browser.

## 📊 Retrieval Evaluation

Retrieval quality is benchmarked using NDCG (Normalized Discounted Cumulative Gain) across three embedding strategies: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, and Google's `gemini-embedding-001`.

- `eval_ndcg.py` — a synthetic test set of 7 clearly distinct topics, used as a sanity check on the evaluation harness itself.
- `eval_ndcg_dbpedia.py` — a harder benchmark using a real query and real relevance judgments from the [DBpedia-Entity v2](https://github.com/iai-group/DBpedia-Entity) test collection (Hasibi et al., SIGIR 2017), where candidate documents share a domain and are genuinely difficult to tell apart.

Full results are in [`eval_results.md`](./eval_results.md). Run either script yourself with:

```bash
python eval_ndcg_dbpedia.py
```

## 🌐 Deployment

Deploy to Streamlit Cloud:
1. Push to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io/)
3. Connect repository
4. Add `GOOGLE_API_KEY` to secrets
5. Deploy!

## 📄 License

MIT License

## 👤 Author

Prisha Chowdhury - [@PrishaChowdhury](https://github.com/PrishaChowdhury)