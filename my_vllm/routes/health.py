"""
健康检查接口
"""

from fastapi import APIRouter, FastAPI, Request

router = APIRouter()


@router.get("/health")
async def health_check(request: Request):
    """服务健康检查"""
    engine_client = request.app.state.engine_client
    return {
        "status": "healthy",
        "engine_errored": engine_client.errored,
        "model": request.app.state.served_model_name,
    }


@router.get("/engine/status")
async def engine_status(request: Request):
    """查看引擎状态（类似 vLLM 的 /engine/status）"""
    return {
        "engine_type": type(request.app.state.engine_client).__name__,
        "model": request.app.state.served_model_name,
        "config": str(request.app.state.vllm_config),
    }


def attach_router(app: FastAPI):
    app.include_router(router)
