import os

from fastapi import APIRouter


DOMAIN = os.getenv("DOMAIN")

router = APIRouter()
if DOMAIN == "ENGLISH":
    from restapi.routes import english

    router.include_router(english.router, tags=["QG_english"])
elif DOMAIN == "HISTORY":
    from restapi.routes import history

    router.include_router(history.router, tags=["QG_history"])
elif DOMAIN == "FILLIN":
    from restapi.routes import fill_in

    router.include_router(fill_in.router, tags=["QG_english_fib"])
elif DOMAIN == "FEEDBACK":
    from restapi.routes import feedback

    router.include_router(feedback.router, tags=["QG_feedback"])
