import asyncio
import json
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from apps.api.core.dependencies import experiments_service_dep, stream_manager_dep

router = APIRouter()


@router.websocket("/ws/summary")
async def ws_summary(ws: WebSocket, svc=Depends(experiments_service_dep), stream=Depends(stream_manager_dep)):
    await ws.accept()
    queue = stream.register()
    try:
        while True:
            # Send latest summary periodically
            summary = await svc.get_experiments_summary()
            await ws.send_json(summary)
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        stream.unregister(queue)


@router.websocket("/ws/experiments")
async def ws_experiments(ws: WebSocket, stream=Depends(stream_manager_dep)):
    await ws.accept()
    queue = stream.register()
    try:
        while True:
            msg = await queue.get()
            await ws.send_json(msg)
    except WebSocketDisconnect:
        stream.unregister(queue)


@router.websocket("/ws/experiments/{experiment_id}")
async def ws_experiment_detail(ws: WebSocket, experiment_id: str, stream=Depends(stream_manager_dep)):
    await ws.accept()
    queue = stream.register()
    try:
        while True:
            msg = await queue.get()
            if msg.get("experiment_id") == experiment_id:
                await ws.send_json(msg)
    except WebSocketDisconnect:
        stream.unregister(queue) 