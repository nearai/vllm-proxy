import os
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .api import router as api_router
from .api.response.response import error, http_exception, ok
from .api.v1.openai import close_http_client, get_http_client
from .logger import log

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    # Startup: initialize the HTTP client pool
    log.info("Initializing HTTP client pool...")
    get_http_client()
    log.info("HTTP client pool initialized")
    yield
    # Shutdown: close the HTTP client pool
    log.info("Closing HTTP client pool...")
    await close_http_client()
    log.info("HTTP client pool closed")


app = FastAPI(lifespan=lifespan)
app.include_router(api_router)


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

    Note: Error logging is intentionally minimal to prevent leaking user data
    (e.g., request payloads, prompts) that may appear in exception messages
    or tracebacks.
    """
    # handle HTTPException
    if isinstance(exc, HTTPException):
        # Log only status code, not details which may contain user data
        log.error(f"HTTPException [{exc.status_code}]")

        safe_messages = {
            400: "Bad request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "Resource not found",
            429: "Too many requests",
            500: "Internal server error",
        }
        generic_message = safe_messages.get(exc.status_code, "An error occurred")

        return http_exception(exc.status_code, generic_message)

    # Log only exception type, not message or traceback to avoid leaking user data
    log.error(f"Unhandled exception [{type(exc).__name__}]")
    # Full details available at DEBUG level only for local debugging
    log.debug(f"Exception details: {traceback.format_exc()}")

    return error(
        status_code=500,
        message="Internal server error",
        type="ServerError",
        param=None,
        code=None,
    )
