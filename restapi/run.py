import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from loguru import logger

from errors.http_error import http_error_handler, validation_exception_handler
from restapi.routes.router import router as api_router
from starlette.middleware.cors import CORSMiddleware


PORT = os.getenv("PORT")
DOMAIN = os.getenv("DOMAIN")

docs_map = {
    "ENGLISH": "/v1/english",
    "HISTORY": "/v1/history",
    "FILLIN": "/v1/english/fillin",
    "FEEDBACK": "/v1/feedback",
}

sys.path.append(Path(__file__).parent.absolute().as_posix())
DOCS_URL = docs_map[DOMAIN]


def get_app() -> FastAPI:
    app = FastAPI(
        title="Question Generation",
        description="APIs for AI-powered Question Generation (AQG)",
        openapi_url=DOCS_URL + "/openapi.json",
    )

    # This middleware enables allow all cross-domain requests to the API from a browser.
    # For production deployments, it could be made more restrictive.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_exception_handler(HTTPException, http_error_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    app.include_router(api_router)

    return app


app = get_app()


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down API")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
