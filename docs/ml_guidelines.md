# ML Service Guidelines (FastAPI + Python)

## Tech Stack
- Python 3.10+
- FastAPI
- Uvicorn
- Pydantic
- Virtual environment (`.venv`)

---

## Project Structure

```
app/
├── main.py
├── routers/
├── services/
├── models/
└── utils/
requirements.txt
```

---

## Required Endpoints

### Health Check
```
GET /health
→ { "status": "OK" }
```

### Analysis
```
POST /analyze

Request:
{
  "texts": ["text1", "text2"]
}

Response:
{
  "sentiments": [...],
  "summary": "..."
}
```

---

## Example FastAPI Pattern

Router:
```python
@router.post("/analyze")
async def analyze(req: AnalysisRequest):
    results = sentiment_service.analyze(req.texts)
    return AnalysisResponse(**results)
```

Dummy service logic:
```python
def analyze(texts):
    sentiments = [{"label": "neutral", "score": 0.5} for _ in texts]
    summary = "Neutral summary placeholder."
    return {"sentiments": sentiments, "summary": summary}
```

---

## AI Usage Rules

When generating ML code:

1. Always follow FastAPI structure (routers, models, services).
2. Only modify files using small patches unless told otherwise.
3. Use Pydantic for request/response models.
4. Keep `.venv` local — do not commit it.
5. Update `requirements.txt` when adding new deps.
6. Avoid heavy ML libs unless explicitly allowed.

---
