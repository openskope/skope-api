from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.api import router, settings


def test_support_build_info_does_not_need_registry_or_job_store(monkeypatch):
    monkeypatch.setattr(settings, "skope_build_id", "abc123")
    monkeypatch.setattr(settings, "environment", "staging")
    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).get("/version")
    assert response.status_code == 200
    assert response.json() == {"build_id": "abc123", "environment": "staging"}
