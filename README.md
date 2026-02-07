# 🏦 Fund Recommendation Chatbot

An AI-powered chatbot that recommends investment funds from a database of **2,700+ VCs, PEs, and investors** based on your criteria.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange.svg)

## ✨ Features

- **Dual Model Support**: Choose between ChatGPT (GPT-4o-mini) or Google Gemini
- **RAG Pipeline**: Fast retrieval using LangChain + FAISS vector database
- **Full Context Mode**: Gemini's 1M token context for complete database access
- **2,700+ Funds**: Comprehensive database of VCs, PEs, Family Offices, and more
- **Modern UI**: Beautiful glassmorphism design with dark mode

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/kartikeyapandey20/Fund-chatbot.git
cd Fund-chatbot
```

### 2. Create virtual environment

```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

Get API keys from:

- OpenAI: https://platform.openai.com/api-keys
- Gemini: https://aistudio.google.com/apikey

### 5. Run the server

```bash
python -m uvicorn main:app --reload --port 8000
```

### 6. Open the chatbot

Navigate to: http://127.0.0.1:8000

## 💬 Usage

1. Select your preferred **Model** (ChatGPT or Gemini)
2. Select **Mode**:
   - **RAG (Fast)**: Retrieves relevant fund chunks for quick responses
   - **Full Context**: Uses entire database (best with Gemini)
3. Ask questions like:
   - _"What VCs invest in fintech at seed stage?"_
   - _"Recommend Series A investors for healthcare startups in India"_
   - _"Which funds focus on deep tech with $1-5M cheque sizes?"_

## 🛠️ Tech Stack

| Component  | Technology                        |
| ---------- | --------------------------------- |
| Backend    | FastAPI, Python                   |
| LLM        | OpenAI GPT-4o-mini, Google Gemini |
| RAG        | LangChain, FAISS                  |
| Frontend   | HTML, CSS, JavaScript             |
| Embeddings | OpenAI Embeddings                 |

## 📁 Project Structure

```
Fund-chatbot/
├── main.py              # FastAPI backend with RAG pipeline
├── requirements.txt     # Python dependencies
├── Funds_databaseV1.csv # Fund database (2,700+ entries)
├── .env                 # API keys (create this)
└── static/
    ├── index.html       # Frontend UI
    ├── style.css        # Styling
    └── script.js        # Frontend logic
```

## 📝 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Pull requests welcome! For major changes, please open an issue first.
