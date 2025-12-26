from typing import Optional

from fastapi.responses import JSONResponse


def ok(data: dict = None):
    return data or dict()


def error(
    status_code: int,
    message: str = "error",
    type: str = "error_type",
    param: str = None,
    code: str = None,
    request_id: Optional[str] = None,
):
    """
    Create an OpenAI-style error response.

    Args:
        status_code: HTTP status code
        message: Human-readable error message
        type: Error type identifier
        param: Optional parameter that caused the error
        code: Optional error code
        request_id: Optional request ID for debugging/support

    Returns:
        JSONResponse with OpenAI-compatible error structure
    """
    error_content = dict(
        message=message,
        type=type,
        param=param,
        code=code,
    )

    # Include request_id in error object if provided
    if request_id:
        error_content["request_id"] = request_id

    content = dict(error=error_content)
    return JSONResponse(status_code=status_code, content=content)


def unexpect_error(context: str = None, error: Exception = None):
    if context is None and error is None:
        message = "An unexpected error occurred."
    elif context and error:
        message = f"{context}: {type(error).__name__}: {str(error)}"
    elif error:
        message = f"An unexpected error occurred: {type(error).__name__}: {str(error)}"
    else:
        message = context

    return error(
        status_code=500,
        message=message,
        type="unknown_error",
        param=None,
        code=None,
    )


def invalid_signing_algo():
    return error(
        status_code=400,
        message="Invalid signing algorithm. Must be 'ed25519' or 'ecdsa'",
        type="invalid_signing_algo",
        param=None,
        code=None,
    )


def http_exception(
    status_code: int, message: str, request_id: Optional[str] = None
):
    return error(
        status_code=status_code,
        message=message,
        type="http_exception",
        request_id=request_id,
    )


def not_found(message: str):
    return error(status_code=404, message=message, type="not_found")