from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import os

# Load environment variables
load_dotenv()

app = FastAPI(title="Fund Recommendation Chatbot")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load and format CSV data at startup
def load_fund_data():
    """Load the funds CSV and format it as context for the LLM."""
    df = pd.read_csv("Funds_databaseV1.csv", encoding='utf-8', on_bad_lines='skip')
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Select key columns for context
    key_columns = [
        'Name', 'Entity Type', 'Stage', 'Sector Focus', 'Sectors Avoided',
        'Typical Cheque Size', 'Min Cheque Size', 'Max Cheque Size',
        'Location', 'Investor Notes', 'Website'
    ]
    
    # Filter to existing columns
    available_columns = [col for col in key_columns if col in df.columns]
    df_filtered = df[available_columns].dropna(subset=['Name'] if 'Name' in available_columns else [])
    
    # Convert to string format for LLM context
    fund_context = df_filtered.to_string(index=False)
    return fund_context

# Load fund data on startup
FUND_CONTEXT = load_fund_data()

# System prompt for the chatbot
SYSTEM_PROMPT = f"""You are a knowledgeable fund recommendation assistant. You have access to a comprehensive database of investment funds, venture capital firms, private equity firms, and other investors.

Based on the user's query, recommend suitable funds from the database below. Consider:
- Investment stage preferences (Seed, Series A, B, C, Growth, Late Stage)
- Sector focus and expertise
- Typical cheque sizes
- Geographic preferences
- Any specific investor notes

When recommending funds:
1. Provide the fund name and key details
2. Explain why each fund is relevant to the user's needs
3. Include cheque size ranges when available
4. Mention sector expertise and stage preferences
5. Be concise but informative

FUND DATABASE:
{FUND_CONTEXT}

Always base your recommendations on the actual fund data provided. If you cannot find suitable matches, explain why and suggest alternative search criteria."""


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process user message and return fund recommendations."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cost-effective model with large context support
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.message}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        return ChatResponse(response=response.choices[0].message.content)
    
    except Exception as e:
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
