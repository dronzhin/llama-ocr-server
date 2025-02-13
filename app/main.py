import os

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import base64
import time
from pydantic import BaseModel
from llama_prompt import new_llama

app = FastAPI()
ollama_url = "http://localhost:11434/v1"

class OCRRequest(BaseModel):
    file: str  # Строка Base64
    engine: str  # Имя движка модели

# Здесь 'directory' должен указывать на папку, где находятся static файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Разрешаем запросы CORS от любого источника
origins = ["*"]  # Для простоты можно разрешить доступ со всех источников
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
   return RedirectResponse(url="/index.html")

@app.get("/index.html", response_class=HTMLResponse)
async def index_page(request: Request):
    # Просто рендерим страницу без параметров engines
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/GetOcr")
async def get_ocr(request: OCRRequest):
    try:
        image_data = base64.b64decode(request.file)
        image = Image.open(BytesIO(image_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при декодировании изображения: {str(e)}")

    if request.engine == "llama3.2:3b":
        start_time = time.time()
        reader = new_llama(url = ollama_url, model= "llama3.2:3b", temp = 0.0, content = text)
        execution_time = time.time() - start_time
        result = {"engine": "easyocr", "execution_time": f"{execution_time:.2f}", "text": reader}

    elif request.engine == "llama3.2-vision":
        start_time = time.time()
        reader = new_llama(url = ollama_url, model= "llama3.2:3b", temp = 0.0, image_url = image, content = text)
        execution_time = time.time() - start_time
        result = {"engine": "tesseract", "execution_time": f"{execution_time:.2f}", "text": reader}

    else:
        raise HTTPException(status_code=400, detail=f"Неподдерживаемый движок OCR: {request.engine}")

    return {"result": result}

@app.get("/GetOcrList")
async def get_ocr_list():
    engines = ["llama3.2:3b", "llama3.2-vision"]
    return {"available_engines": engines}

if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get('PORT', 8000)))