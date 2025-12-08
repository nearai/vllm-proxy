import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .api import router as api_router
from .api.response.response import ok, error, http_exception
from .logger import log

app = FastAPI()
app.include_router(api_router)

GIT_REV_PATH = "/etc/.GIT_REV"


def _read_git_rev() -> str:
    """Read git revision from /etc/.GIT_REV file."""
    try:
        if os.path.exists(GIT_REV_PATH):
            with open(GIT_REV_PATH, "r") as f:
                return f.read().strip()
    except Exception:
        pass
    return "unknown"


GIT_REV = _read_git_rev()


@app.get("/")
async def root():
    return ok()


@app.get("/version")
async def show_version():
    """Return version information for this vllm-proxy instance."""
    ver = {"version": GIT_REV, "type": "proxy"}
    return JSONResponse(content=ver)


# Custom global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Handle all uncaught exceptions globally.
    """
    # handle HTTPException
    if isinstance(exc, HTTPException):
        log.error(f"HTTPException: status={exc.status_code}")
        return http_exception(exc.status_code, exc.detail)

    log.error(f"Unhandled exception: {type(exc).__name__}")
    return error(
        status_code=500,
        message=str(exc),
        type=type(exc).__name__,
        param=None,
        code=None,
    )
