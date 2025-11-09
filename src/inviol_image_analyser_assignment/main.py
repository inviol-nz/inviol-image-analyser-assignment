from __future__ import annotations

import logging
import sys

from fastapi import FastAPI

from .api.routes import analyse_router
from .config.settings import settings
from .core.cache import AnalysisCache
from .core.errors import setup_exception_handlers
from .services.cv.detection_service import ObjectDetector
from .services.rule_engine import RuleEngine

logger = logging.getLogger("inviol_app")

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(levelname)s [%(name)s] %(message)s",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)
logger.propagate = False


def create_app() -> FastAPI:

    app = FastAPI(
        title="Inviol Image Analyser",
        description="Service for workplace health & safety image analysis.",
        version="0.2.0",
    )

    # Exception handlers
    setup_exception_handlers(app)

    # Routers
    app.include_router(analyse_router)

    # Startup initialisation of heavy services
    @app.on_event("startup")
    def on_startup() -> None:
        logger.info("Starting app in environment: %s", settings.environment)

        detector = ObjectDetector()
        rule_engine = RuleEngine.default()
        cache = AnalysisCache(settings.cache_size) if settings.cache_enabled else None

        app.state.detector = detector
        app.state.rule_engine = rule_engine
        app.state.cache = cache

    return app


app = create_app()
