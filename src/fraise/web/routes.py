"""Catalog, generation, and training operations for the local API."""

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from .configuration import describe_architectures, training_defaults, validate_training
from .generation import GenerationStream
from .schemas import GenerationRequest, TokenizeRequest, TrainingRequest
from .tokenization import TokenizationWorkbench


def create_routes(catalog, gate, inference, jobs, device_info):
    router = APIRouter(prefix="/api")
    workbench = TokenizationWorkbench(catalog)

    @router.get("/catalog")
    def get_catalog():
        from fraise.architectures import get_architecture
        from fraise.data.sources import SOURCE_NAMES
        from fraise.training.run.manifest import OPERATIONAL_SETTINGS

        return {
            "architectures": describe_architectures(),
            "tokenizers": catalog.tokenizers(),
            "sources": list(SOURCE_NAMES),
            "model_defaults": dict(get_architecture("transformer").example_config),
            "training_defaults": training_defaults(),
            "operational_settings": sorted(OPERATIONAL_SETTINGS),
            "device": device_info,
        }

    @router.get("/runs")
    def get_runs():
        return {"runs": catalog.runs()}

    @router.get("/status")
    def get_status():
        return {"busy": gate.busy}

    @router.post("/config/validate")
    def validate(request: TrainingRequest):
        normalized, _ = validate_training(request, catalog, device_info["resolved"])
        return normalized

    @router.post("/generate")
    def generate(request: GenerationRequest):
        references = [(ref, *catalog.resolve(ref)) for ref in request.models]
        stream = GenerationStream(request, references, inference, gate)
        return StreamingResponse(
            stream.events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/hierarchies")
    def get_hierarchies():
        return {"hierarchies": workbench.hierarchy_files()}

    @router.post("/tokenize")
    def tokenize(request: TokenizeRequest):
        return workbench.analyze(request)

    @router.get("/jobs")
    def get_jobs():
        return {"jobs": jobs.list()}

    @router.post("/jobs", status_code=201)
    def start_job(request: TrainingRequest):
        return jobs.start(request)

    @router.get("/jobs/{identifier}")
    def get_job(identifier: str, cursor: int = Query(default=0, ge=0)):
        return jobs.get(identifier, cursor)

    @router.post("/jobs/{identifier}/stop")
    def stop_job(identifier: str):
        return jobs.stop(identifier)

    return router
