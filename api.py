from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from models.chat_request_model import ChatRequest
from services.chat_service import ChatService
from services.upload_file_service import UploadFileService


app = FastAPI()

# Allow CORS for local development (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "data", "uploads")
# os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/upload/")
async def upload_file(file: UploadFile = File(...)):
    uploader = UploadFileService()
    try:
        entry = await uploader.save_upload(file)
        return JSONResponse(content={"message": "Uploaded", "file": entry})
    except FileExistsError as e:
        return JSONResponse(status_code=400, content={"message": str(e)})


@app.post("/api/files/")
async def get_files():
    uploader = UploadFileService()
    response = await uploader.get_file_names()
    return JSONResponse(content=response)

# New chat endpoint for user messages


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("VALIDATION ERROR:", exc.errors())   # <-- check your server console
    return PlainTextResponse(str(exc), status_code=422)

@app.post("/api/chat/")
async def chat_endpoint(req: ChatRequest):
    chat_service = ChatService()
    messages = [m.dict() for m in req.history]
    messages.append({"role": "user", "content": req.message})
    try:
        llm_reply = await chat_service.chat(messages)
    except Exception as e:
        llm_reply = f"Error communicating with LLM: {str(e)}"
    response = {
        "reply": llm_reply,
        "history": messages
    }
    return JSONResponse(content=response)


@app.get("/api/home/")
async def home():
    retriever_tool = chat_service.get_retriever_tool()
    response = retriever_tool.invoke({"query": "what is langchain?"})
    response = {
        "message": response["result"]
    }
    return JSONResponse(content=response)
