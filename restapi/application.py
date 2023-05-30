import logging
import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from environs import Env
from starlette.middleware.cors import CORSMiddleware


env = Env()
env.read_env(f"{os.getcwd()}/.env")

logging.basicConfig(
    format="%(asctime)s %(levelname)-5s - %(message)s ", datefmt="%m/%d/%Y %I:%M:%S %p"
)
logger = logging.getLogger("questgen_demo")
logging.getLogger("questgen_demo").setLevel(logging.INFO)


def get_application() -> FastAPI:
    application = FastAPI(title="QuestGen Demo", debug=True, version="v.01")

    origins = ["*"]

    application.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.mount(
        "/questgen", StaticFiles(directory="app/QuestGen"), name="questgen"
    )
    return application


app = get_application()


@app.get("/", response_class=HTMLResponse)
async def demo():
    # with open("QuestGen/webapp.html", "r") as f:
    #     return "".join(f.readlines())
    with open("app/QuestGen/webapp.html", "r") as f:
        return "".join(f.readlines())


if __name__ == "__main__":
    uvicorn.run(app, debug=False, host="0.0.0.0", port=5051)
