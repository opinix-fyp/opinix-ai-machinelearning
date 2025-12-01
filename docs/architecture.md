# Opinix System Architecture

## Overview
Opinix is an AI-powered community polling and sentiment analysis platform.  
The system is split into three separate repositories:

1. **opinix-backend**  
   - Java + Spring Boot  
   - Core business logic, API gateway, database operations  
   - Communicates with the ML service

2. **opinix-frontend**  
   - React + TypeScript + Vite  
   - User interface for creating polls, viewing analytics, and interacting with the system  
   - Sends API calls to the backend

3. **opinix-ml-service**  
   - Python + FastAPI  
   - Handles sentiment analysis, text classification, and NLP tasks  
   - Exposes ML endpoints that the backend calls

---

## High-Level Data Flow

```
Frontend (React)
     ↓ REST API
Backend (Spring Boot)
     ↓ HTTP Request
ML Service (FastAPI)
     ↓ ML Processing
Backend → Frontend → User
```

---

## Responsibilities of Each Layer

### Frontend
- Creates and displays polls
- Sends user responses to backend
- Displays sentiment and poll analytics
- Interacts only with **backend**, never ML directly

### Backend
- Defines REST API routes
- Stores polls, responses, metadata
- Calls ML service for analysis
- Combines ML results with DB
- Returns structured JSON to frontend

### ML Service
- Receives raw text responses
- Performs sentiment analysis
- Returns sentiment + summary
- Future: topics, keyword extraction, clustering

---

## Backend ↔ ML API Contract

### Request (Backend → ML)
```json
{
  "texts": ["example text 1", "example text 2"]
}
```

### Response (ML → Backend)
```json
{
  "sentiments": [
    { "label": "positive", "score": 0.92 },
    { "label": "negative", "score": 0.78 }
  ],
  "summary": "Overall feedback is mixed."
}
```

---

## Development Rules

- All coding is done in the **dev** branch.
- `main` is protected — no direct commits.
- Frontend → Backend → ML routing only (no shortcuts).
- All layers must follow their respective guidelines:
  - Spring controller/service/repository pattern
  - React + TypeScript with clean structure
  - FastAPI router/service/model pattern

---

## AI Usage Rules (for any AI assistant)

When generating code for Opinix:

1. **Follow this architecture exactly.**
2. Do NOT introduce new libraries unless explicitly allowed.
3. Prefer patches/diffs, not full rewrites.
4. Respect each repo’s folder structure.
5. All analysis must flow Backend → ML Service.
6. Frontend must NEVER call the ML service directly.
7. Use TypeScript on frontend.
8. Use DTOs and Services on backend.
9. Use Pydantic models on ML service.

---
