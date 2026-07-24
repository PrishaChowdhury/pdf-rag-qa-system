# PDF RAG Question Answering System

A Retrieval-Augmented Generation (RAG) system that allows you to ask questions about PDF documents using AI.

## 🚀 Features

- Upload PDF documents
- Ask questions in natural language
- Get AI-powered answers from your documents
- Built with Streamlit, LangChain, and Google Gemini

## 🛠️ Tech Stack

- **LLM**: Google Gemini 1.5 Flash
- **Embeddings**: HuggingFace (sentence-transformers)
- **Vector Database**: FAISS
- **Framework**: LangChain
- **UI**: Streamlit

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/rag-project.git
cd rag-project
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

Prisha Chowdhury - [@yPrishaChowdhury](https://github.com/PrishaChowdhury)