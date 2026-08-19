from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .views import atlas, classify, intervene, logit_lens

app = FastAPI(title="vitlab Explorer", version="0.1.0")

# Dev: Vite serves on 5173 and proxies /api -> 8000, but allow direct CORS too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(classify.router, prefix="/api", tags=["classify"])
app.include_router(atlas.router, prefix="/api", tags=["atlas"])
app.include_router(intervene.router, prefix="/api", tags=["intervene"])
app.include_router(logit_lens.router, prefix="/api", tags=["logit-lens"])


@app.get("/api/health")
def health():
    from . import state
    return {"status": "ok", "device": state.DEVICE}
