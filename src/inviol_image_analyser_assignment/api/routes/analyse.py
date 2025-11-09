from __future__ import annotations

import io
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from inviol_image_analyser_assignment.api.dependencies import require_api_key
from inviol_image_analyser_assignment.config.settings import settings
from inviol_image_analyser_assignment.core.cache import AnalysisCache
from inviol_image_analyser_assignment.models import AnalysisResult
from inviol_image_analyser_assignment.services.cv import ObjectDetector
from inviol_image_analyser_assignment.services.rule_engine import RuleEngine

router = APIRouter()


def _get_services(
    request: Request,
) -> tuple[ObjectDetector, RuleEngine, AnalysisCache[AnalysisResult] | None]:
    try:
        detector: ObjectDetector = request.app.state.detector
        rule_engine: RuleEngine = request.app.state.rule_engine
        cache: AnalysisCache[AnalysisResult] | None = request.app.state.cache
    except AttributeError as exc:
        raise RuntimeError("Services not initialised") from exc
    return detector, rule_engine, cache


@router.get("/healthcheck", tags=["health"])
async def get_healthcheck() -> dict[str, str]:
    return {"status": "healthy"}


@router.post(
    "/analyse",
    response_model=AnalysisResult,
    status_code=status.HTTP_200_OK,
    summary="Analyse an image for health & safety risks",
    tags=["analysis"],
    dependencies=[Depends(require_api_key)]
)
async def post_analyse(
    request: Request,
    file: Annotated[UploadFile, File(description="JPEG or PNG image")],
) -> AnalysisResult:
    """
    Accept an image upload, run object detection, apply safety rules, and
    return a structured risk assessment.

    Validation:
    - Accepts multipart/form-data with a single `file` field.
    - Only JPEG/PNG content types are allowed.
    - Maximum file size is configured via MAX_FILE_SIZE_BYTES.
    """

    if file.content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "invalid_content_type",
                "message": (
                    f"Unsupported content type: {file.content_type}. "
                    f"Allowed types: {', '.join(settings.allowed_content_types)}."
                ),
            },
        )

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "invalid_file_extension",
                "message": (
                    f"Unsupported file extension: '{suffix or 'none'}'. "
                    "Only .jpg, .jpeg and .png files are accepted."
                ),
            },
        )

    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "empty_file",
                "message": "Empty file uploaded.",
            },
        )
    if len(raw_bytes) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "file_too_large",
                "message": (
                    f"File too large. Maximum size is "
                    f"{settings.max_file_size_bytes} bytes."
                ),
            },
        )

    detector, rule_engine, cache = _get_services(request)

    # Cache lookup
    if cache is not None:
        cached = cache.get(raw_bytes)
        if cached is not None:
            return cached

    # Image decoding / corruption handling
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "invalid_image",
                "message": "Uploaded file is not a valid JPEG/PNG image.",
            },
        ) from exc

    # Run heavy detection+rules in a threadpool
    def _run_analysis() -> AnalysisResult:
        detections, image_width, image_height = detector.detect(image=image)
        overall_score, risk_level, risk_rating, violations = rule_engine.evaluate(
            detections=detections,
            image_width=image_width,
            image_height=image_height,
        )
        return AnalysisResult(
            risk_rating=risk_rating,
            overall_risk_score=overall_score,
            risk_level=risk_level,
            model_name=detector.model_name,
            model_version=detector.model_version,
            detections=detections,
            violations=violations,
        )

    analysis_result: AnalysisResult = await run_in_threadpool(_run_analysis)

    if cache is not None:
        cache.set(raw_bytes, analysis_result)

    return analysis_result
