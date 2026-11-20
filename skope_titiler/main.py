from titiler.application.main import app

from skope_titiler.telemetry import RequestTimingMiddleware

app.add_middleware(RequestTimingMiddleware, service="titiler")
