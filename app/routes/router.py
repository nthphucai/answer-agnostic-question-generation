from fastapi import APIRouter

from app.routes import generation, home


router = APIRouter()

router.include_router(home.router, tags=["home"])
router.include_router(generation.router, tags=["generation"])
