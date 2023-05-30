from fastapi import HTTPException, status
from fastapi.exceptions import RequestValidationError

from starlette.requests import Request
from starlette.responses import JSONResponse


async def http_error_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(content={"errors": [exc.detail]}, status_code=exc.status_code)


async def validation_exception_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        content={"error": f"Invalid input parameters supplied : {exc.errors()}"},
        status_code=status.HTTP_400_BAD_REQUEST,
    )
