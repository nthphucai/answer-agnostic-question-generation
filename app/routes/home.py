import fastapi
from fastapi.responses import HTMLResponse


router = fastapi.APIRouter()


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    with open("app/webapp/webapp.html", "r") as f:
        return "".join(f.readlines())


@router.get("/tutorial", response_class=HTMLResponse, include_in_schema=False)
async def tutorial():
    with open("app/webapp/pages/tutorial.html", "r") as f:
        return "".join(f.readlines())
