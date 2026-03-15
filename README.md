# Opinix ML Service

Python FastAPI service for backend-only ML inference in Opinix.

## Architecture

Repositories:
1. opinix-backend (Java + Spring Boot)
2. opinix-frontend (React + TypeScript + Vite)
3. opinix-ml-service (Python + FastAPI)

Request flow:
Frontend -> Backend -> ML Service

The frontend must not call this service directly.

## Project Structure

app/
- main.py
- routers/
- services/
- models/
- utils/

## Setup

Python 3.10+

PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

## Run

uvicorn app.main:app --reload

Optional compatibility entrypoint:

uvicorn api:app --reload

## API Contract

### GET /health

Response:

{
  "status": "OK"
}

### POST /analyze

Request:

{
  "texts": ["text1", "text2"]
}

Response:

{
  "sentiments": [
    { "label": "positive", "score": 0.92 },
    { "label": "negative", "score": 0.78 }
  ],
  "summary": "Overall feedback is mixed."
}

## Notes

- Current implementation wraps the trained model in saved_model for inference.
- Service logic is separated from routing for easy extension.
- Keep .venv local and do not commit it.
