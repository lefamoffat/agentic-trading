from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from apps.api.core.dependencies import experiments_service_dep
from apps.api.schemas import ExperimentLaunchRequest, ExperimentLaunchResponse
import subprocess, sys, json, uuid, os
from fastapi import BackgroundTasks
from src.training import get_training_service

router = APIRouter()


@router.get("", summary="List recent experiments")
async def list_experiments(limit: int = 20, svc=Depends(experiments_service_dep)) -> List[dict]:
    return await svc.get_recent_experiments(limit=limit)


@router.get("/summary", summary="System KPI summary")
async def experiments_summary(svc=Depends(experiments_service_dep)) -> dict:
    return await svc.get_experiments_summary()


# ------------------------------------------------------------------
# NOTE: Summary route must appear *before* the dynamic /{experiment_id}
#       route, otherwise it will be captured as a path parameter.
# ------------------------------------------------------------------


@router.get("/{experiment_id}", summary="Get experiment details")
async def experiment_detail(experiment_id: str, svc=Depends(experiments_service_dep)) -> dict:
    data = await svc.get_experiment_details(experiment_id)
    if not data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")
    return data


# ------------------------------------------------------------------
# Launch / stop helpers
# ------------------------------------------------------------------


@router.post("/", response_model=ExperimentLaunchResponse, status_code=202)
async def launch_experiment(body: ExperimentLaunchRequest, background: BackgroundTasks):
    """Launch a new training experiment (detached worker)."""

    experiment_id = f"api_{uuid.uuid4().hex[:8]}"
    config = body.dict()

    # Path to worker script
    worker_script = os.path.join("scripts", "training", "run_training_worker.py")

    cmd = [
        "uv",
        "run",
        sys.executable,
        worker_script,
        experiment_id,
        json.dumps(config),
    ]

    def _spawn():
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    background.add_task(_spawn)

    return ExperimentLaunchResponse(experiment_id=experiment_id, status="starting")


@router.post("/{experiment_id}/stop", status_code=202)
async def stop_experiment(experiment_id: str):
    """Request to stop a running experiment via training service."""
    training_service = await get_training_service()
    res = await training_service.stop_experiment(experiment_id)
    if res.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"status": "stopping"} 