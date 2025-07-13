import orjson
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from core.service_factories import get_recommendation_manager, get_user_profile_service
from managers.recommendation_manager import RecommendationManager
from schemas.filters import FiltersResponse
from schemas.recommendation import RecommendationRequest, RecommendationResponse
from services.user_profile_service import UserProfileService

router = APIRouter()


@router.get("/profiles/{user_id}")
async def get_profile(user_id: str, profile_service: UserProfileService = Depends(get_user_profile_service)):
    profile = await profile_service.get_profile(user_id)
    if not profile:
        return {"error": "Profile not found"}
    return profile


@router.post("/upload-watch-history")
async def upload_watch_history(
    user_id: str, file: UploadFile = File(...), profile_service: UserProfileService = Depends(get_user_profile_service)
):
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only JSON files are supported")
    try:
        content = await file.read()
        data = orjson.loads(content)
        if isinstance(data, list):
            movies = data
        elif isinstance(data, dict) and "movies" in data:
            movies = data["movies"]
        else:
            raise ValueError("File must be a list of movies or contain a 'movies' array")

        await profile_service.create_and_save_profile(user_id, movies)

        return {"message": f"Watch history uploaded successfully for user {user_id}", "movies_count": len(movies)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(
    request: RecommendationRequest,
    manager: RecommendationManager = Depends(get_recommendation_manager),
):
    return await manager.recommend(request)


@router.get("/filters", response_model=FiltersResponse)
def get_filters():
    return FiltersResponse(
        genres=["Action", "Drama", "Sci-Fi", "Crime"],
        years=list(range(1970, 2024)),
        durations=[90, 120, 150, 180],
        countries=["USA", "UK", "FR", "JP"],
    )
