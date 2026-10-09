# DocuMind

Chat with your PDFs. DocuMind is a simple RAG (retrieval-augmented generation) app that runs fully on your own laptop. No OpenAI key, no cloud cost, and your documents never leave your machine.

## Features

- Ask questions about your PDF files in a clean light-themed chat UI
- Answers stream word by word, with live progress while it searches
- Shows the PDF name and page number for every answer
- Upload PDFs and rebuild the index from the sidebar
- Optional request tracing with LangSmith

## Tech used

| Tool | Purpose |
|---|---|
| Python | Main language |
| Streamlit | Web app and chat UI |
| LangChain | Connects the pieces together |
| FAISS | Local vector database |
| Ollama | Runs the AI models locally |
| Qwen 2.5 (1.5B) | Model that writes answers |
| nomic-embed-text | Model that turns text into searchable numbers |
| LangSmith | Optional tracing |

## How it works

```
PDFs → split into chunks → embeddings → FAISS index
Question → find closest chunks → Qwen writes answer from them
```

## Prerequisites

- Python 3.9 or later
- [Ollama](https://ollama.com) installed and running

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/<your-username>/documind.git
cd documind
```

**2. Download the models**

```bash
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
```

**3. Create and activate a virtual environment**

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

**4. Install packages**

```bash
pip install -r requirements.txt
```

**5. (Optional) Turn on LangSmith tracing**

Create a `.env` file in the project folder:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=documind
```

## Run

```bash
streamlit run documind_app.py
```

Then:

1. Upload PDFs in the sidebar, or copy them into the `pdfs` folder
2. Click **Update index** (the first build takes a few minutes)
3. Ask your questions in the chat box

The index is saved in `faiss_index`, so the next start is fast. Click **Update index** again whenever you add or change PDFs.

## Project structure

```
documind/
├── documind_app.py        # The whole app
├── requirements.txt       # Python packages
├── .streamlit/
│   └── config.toml        # Light theme
├── pdfs/                  # Your PDF files (not uploaded to Git)
├── faiss_index/           # Saved search index (not uploaded to Git)
├── .env                   # Your keys (not uploaded to Git)
├── .gitignore
└── README.md
```

## Settings

All settings are at the top of `documind_app.py`:

| Setting | What it does |
|---|---|
| `APP_NAME` | Name shown in the app |
| `LLM_MODEL` | Answer model. Try `qwen2.5:7b` for better answers |
| `EMBED_MODEL` | Search model |
| `CHUNK_SIZE` | Size of each text piece. Bigger means fewer chunks and a faster build |
| `CHUNK_OVERLAP` | Shared text between neighbouring chunks |

The sidebar slider **Chunks to use** controls how much text is sent to the model for each question.

## Troubleshooting

| Problem | Fix |
|---|---|
| Connection error to Ollama | Start it with `ollama serve` |
| Model not found | Run the `ollama pull` commands above |
| "No text found" | Your PDFs are scanned images and need OCR first |
| Wrong or odd answers | Use a bigger model, or raise **Chunks to use** |
| Changed the embedding model | Delete the `faiss_index` folder and click **Update index** |
| Theme not changing | Restart Streamlit. A browser refresh is not enough |
| No traces in LangSmith | Check the keys in `.env` and that tracing is `true` |

## Limitations

- Each question is answered on its own. The app doesn't remember earlier chat messages.
- Small local models can miss details. Answers are weaker than big cloud models.
- Scanned PDFs are not supported.
