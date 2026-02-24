from fastapi import FastAPI
from pydantic import BaseModel
from llm import get_ai_response
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    province: str
    city: str
    session_id: str

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        stream = get_ai_response(
            user_message=req.message,
            user_province=req.province,
            user_city=req.city,
            session_id=req.session_id
        )

        result = ""

        for chunk in stream:
            try:
                result += chunk.content
            except:
                result += str(chunk)


        return {"response": result}

    except Exception as e:
        return {"response": f"ERROR: {str(e)}"}
