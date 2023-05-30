import fastapi


router = fastapi.APIRouter()

# @router.get("/", include_in_schema=True)
# async def index():
#     return {"message": "Hello! Use /docs for more detail about API!"}


@router.get("/", include_in_schema=False)
async def index():
    return "API is serving..."
