from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import os
import logging
import traceback

# LangChain imports for RAG
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Configure logging with enhanced formatting for Docker visibility
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler()  # Ensures output to stdout for Docker logs
    ]
)

# Set third-party loggers to WARNING to reduce noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="Fund Recommendation Chatbot")


# Global exception handler to catch all unhandled errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions and log them."""
    logger.error(f"Unhandled exception on {request.method} {request.url}")
    logger.error(f"Exception type: {type(exc).__name__}")
    logger.error(f"Exception message: {str(exc)}")
    logger.error(f"Full traceback:\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


@app.on_event("startup")
async def startup_event():
    """Log when the application starts."""
    logger.info("="*60)
    logger.info("Fund Recommendation Chatbot starting up...")
    logger.info(f"OpenAI API Key configured: {'Yes' if openai_api_key else 'No'}")
    logger.info(f"Gemini API Key configured: {'Yes' if gemini_api_key else 'No'}")
    logger.info(f"Gemini client initialized: {'Yes' if gemini_client else 'No'}")
    logger.info(f"Vector store initialized: {'Yes' if VECTOR_STORE else 'No'}")
    logger.info(f"Funds loaded: {len(FUND_DF)} records")
    logger.info("="*60)


@app.on_event("shutdown")
async def shutdown_event():
    """Log when the application shuts down."""
    logger.info("Fund Recommendation Chatbot shutting down...")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500", "http://127.0.0.1:8000", "http://localhost:8000","https://fund-chatbot-564206280112.europe-west1.run.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)

# Initialize Gemini client
gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_client = None
try:
    from google import genai
    if gemini_api_key:
        gemini_client = genai.Client(api_key=gemini_api_key)
        logger.info("Gemini client initialized successfully")
except ImportError:
    logger.warning("Google GenAI package not available")
except Exception as e:
    logger.warning(f"Failed to initialize Gemini client: {e}")


def load_fund_data():
    """Load the funds CSV and format it as context for the LLM."""
    df = pd.read_csv("Funds_databaseV1.csv", encoding='utf-8', on_bad_lines='skip', skiprows=1)
    df.columns = df.columns.str.strip()
    
    # Select key columns for context
    key_columns = [
        'Name', 'Entity Type', 'Stage', 'Sector Focus', 'Sectors Avoided',
        'Typical Cheque Size', 'Min Cheque Size', 'Max Cheque Size',
        'Location', 'Investor Notes', 'Website'
    ]
    
    available_columns = [col for col in key_columns if col in df.columns]
    df_filtered = df[available_columns].dropna(subset=['Name'] if 'Name' in available_columns else [])
    
    return df_filtered


def create_fund_documents(df):
    """Convert DataFrame rows to text documents for RAG."""
    documents = []
    for _, row in df.iterrows():
        doc_parts = []
        for col in df.columns:
            value = row[col]
            if pd.notna(value) and str(value).strip():
                doc_parts.append(f"{col}: {value}")
        if doc_parts:
            documents.append("\n".join(doc_parts))
    return documents


def setup_rag_pipeline(documents):
    """Set up FAISS vector store with chunked documents."""
    logger.info(f"Setting up RAG pipeline with {len(documents)} documents")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    
    chunks = []
    for doc in documents:
        splits = text_splitter.split_text(doc)
        chunks.extend(splits)
    
    logger.info(f"Created {len(chunks)} chunks from documents")
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    logger.info("FAISS vector store created successfully")
    return vector_store


# Load data and set up RAG at startup
logger.info("Loading fund data...")
FUND_DF = load_fund_data()
FUND_DOCUMENTS = create_fund_documents(FUND_DF)
FUND_CONTEXT = FUND_DF.to_string(index=False)
logger.info(f"Loaded {len(FUND_DF)} funds, context length: {len(FUND_CONTEXT)} chars")

# Set up RAG vector store
logger.info("Setting up RAG vector store (this may take a moment)...")
try:
    VECTOR_STORE = setup_rag_pipeline(FUND_DOCUMENTS)
except Exception as e:
    logger.error(f"Failed to set up RAG pipeline: {e}")
    VECTOR_STORE = None


# System prompt template
SYSTEM_PROMPT_TEMPLATE = """You are a knowledgeable fund recommendation assistant. You have access to a database of investment funds, venture capital firms, private equity firms, and other investors.

Based on the user's query, recommend suitable funds from the context provided. Consider:
- Investment stage preferences (Seed, Series A, B, C, Growth, Late Stage)
- Sector focus and expertise
- Typical cheque sizes
- Geographic preferences

When recommending funds:
1. Provide the fund name and key details
2. Explain why each fund is relevant to the user's needs
3. Include cheque size ranges when available
4. Mention sector expertise and stage preferences
5. Be concise but informative

{context_section}

Always base your recommendations on the actual fund data provided. If you cannot find suitable matches, explain why and suggest alternative search criteria."""


class ChatRequest(BaseModel):
    message: str
    model: str = "chatgpt"  # "chatgpt" or "gemini"
    mode: str = "rag"  # "rag" or "full_context"


class ChatResponse(BaseModel):
    response: str
    model_used: str
    mode_used: str


def get_rag_context(query: str, k: int = 10) -> str:
    """Retrieve relevant context from vector store."""
    if VECTOR_STORE is None:
        return ""
    
    docs = VECTOR_STORE.similarity_search(query, k=k)
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    return context


def chat_with_openai(message: str, context: str) -> str:
    """Send message to OpenAI and get response."""
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        context_section=f"FUND DATABASE CONTEXT:\n{context}"
    )
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ],
        temperature=0.7,
        max_tokens=1500
    )
    
    return response.choices[0].message.content


def chat_with_gemini(message: str, context: str) -> str:
    """Send message to Gemini and get response."""
    if gemini_client is None:
        raise ValueError("Gemini client not initialized. Please set GEMINI_API_KEY in .env file.")
    
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        context_section=f"FUND DATABASE CONTEXT:\n{context}"
    )
    
    full_prompt = f"{system_prompt}\n\nUser Query: {message}"
    
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=full_prompt,
    )
    
    return response.text


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process user message and return fund recommendations."""
    try:
        model = request.model.lower()
        mode = request.mode.lower()
        
        logger.info(f"Chat request - Model: {model}, Mode: {mode}, Message: {request.message[:100]}...")
        
        # Get context based on mode
        if mode == "rag":
            context = get_rag_context(request.message)
            logger.info(f"RAG context retrieved: {len(context)} chars")
        else:
            # Full context mode
            if model == "gemini":
                # Gemini can handle full context (1M tokens)
                context = FUND_CONTEXT
                logger.info(f"Using full context for Gemini: {len(context)} chars")
            else:
                # ChatGPT has smaller context, fall back to RAG
                logger.warning("Full context too large for ChatGPT, falling back to RAG")
                context = get_rag_context(request.message, k=20)  # Get more chunks
                mode = "rag (fallback)"
        
        # Get response from selected model
        if model == "gemini":
            response_text = chat_with_gemini(request.message, context)
        else:
            response_text = chat_with_openai(request.message, context)
        
        return ChatResponse(
            response=response_text,
            model_used=model,
            mode_used=mode
        )
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
