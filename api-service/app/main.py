
from fastapi import FastAPI
from app.schema import ChatRequest
from app.triton_client import infer

app = FastAPI()

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/chat")
def chat(request: ChatRequest):

    result = infer()

    return {
        "user_input": request.text,
        "model_output": result.tolist()
    }
