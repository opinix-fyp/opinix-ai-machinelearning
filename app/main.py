from fastapi import FastAPI

app = FastAPI(title = "Opinix ML Service")

@app.get("/health")
async def health_check():
    return {"status": "OK", "message": "ML service is running"}