from fastapi import FastAPI, UploadFile
from .models import AnalysisResult

app = FastAPI()

MAX_FILE_SIZE_MB = 5
ALLOWED_TYPES = {"jpeg", "png", "jpg"}

@app.get("/healthcheck")
async def get_healthcheck():
    return {"status": "healthy"}


@app.post("/analyse")
async def post_analyse(file: UploadFile) -> AnalysisResult:
    # TODO: Implement the actual image analysis logic


    print(f"Received file: {file.filename}")
    return AnalysisResult(risk_rating=5)
