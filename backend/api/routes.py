import json
import os
import tempfile
from typing import Dict

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from core.recommendation_factory import get_recommendation_manager
from managers.recommendation_manager import RecommendationManager
from repositories.user_profile import UserProfileRepository
from schemas.filters import FiltersResponse
from schemas.recommendation import RecommendationRequest, RecommendationResponse

router = APIRouter()

# Temporary storage for uploaded files (in production, use Redis or database)
uploaded_files: Dict[str, str] = {}


@router.get("/profiles/{user_id}")
async def get_profile(user_id: str, session=Depends(get_recommendation_manager)):
    repo = UserProfileRepository(session)
    profile = await repo.get_by_id(user_id)
    if not profile:
        return {"error": "Not found"}
    return profile


@router.post("/upload-watch-history")
async def upload_watch_history(user_id: str, file: UploadFile = File(...)):
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only JSON files are supported")
    try:
        content = await file.read()
        data = json.loads(content.decode("utf-8"))
        if isinstance(data, list):
            movies = data
        elif isinstance(data, dict) and "movies" in data:
            movies = data["movies"]
        else:
            raise ValueError("File must be a list of movies or contain a 'movies' array")
        uploaded_files[user_id] = content.decode("utf-8")
        return {"message": f"Watch history uploaded successfully for user {user_id}", "movies_count": len(movies)}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(
    request: RecommendationRequest,
    manager: RecommendationManager = Depends(get_recommendation_manager),
):
    user_id = request.user_id or "demo"
    file_content = uploaded_files.get(user_id)
    return await manager.recommend(request, file_content)


@router.get("/filters", response_model=FiltersResponse)
def get_filters():
    return FiltersResponse(
        genres=["Action", "Drama", "Sci-Fi", "Crime"],
        years=list(range(1970, 2024)),
        durations=[90, 120, 150, 180],
        countries=["USA", "UK", "FR", "JP"],
    )
