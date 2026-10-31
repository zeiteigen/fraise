"""Compose the local application and own its resources."""

from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from fraise.paths import data_root

from .catalog import Catalog
from .generation import InferenceSession
from .jobs import TrainingJobs
from .routes import create_routes
from .state import BusyError, ComputeGate


def create_app(*, data_dir=None, models_dir=None, device=None):
    available = torch.cuda.is_available()
    resolved = (
        device if device and device != "auto" else ("cuda" if available else "cpu")
    )
    if resolved not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if resolved == "cuda" and not available:
        raise ValueError("CUDA was requested but is unavailable")
    catalog = Catalog(data_root(data_dir), models_dir)
    gate, inference = ComputeGate(), InferenceSession(resolved)
    jobs = TrainingJobs(catalog, gate, inference, resolved)

    @asynccontextmanager
    async def lifespan(app):
        yield
        import asyncio

        await asyncio.to_thread(jobs.close)
        inference.unload()

    app = FastAPI(title="Fraise", lifespan=lifespan)
    app.state.catalog, app.state.gate = catalog, gate
    app.state.inference, app.state.jobs = inference, jobs
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"]
    )

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
            return JSONResponse(
                {"detail": "Use the local Fraise page to access this API"},
                status_code=403,
            )
        return await call_next(request)

    @app.exception_handler(Exception)
    async def unexpected(request, error):
        return JSONResponse({"detail": str(error)}, status_code=500)

    async def expected(request, error):
        code = (
            409
            if isinstance(error, BusyError)
            else (404 if isinstance(error, FileNotFoundError) else 400)
        )
        return JSONResponse({"detail": str(error)}, status_code=code)

    for error_type in (BusyError, ValueError, TypeError, OSError):
        app.add_exception_handler(error_type, expected)
    device_info = {
        "requested": device or "auto",
        "resolved": resolved,
        "cuda_available": available,
    }
    app.include_router(create_routes(catalog, gate, inference, jobs, device_info))
    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(static / "index.html")

    return app
