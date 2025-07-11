from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from schemas.watch_history import MovieHistoryItem
from schemas.recommendation import RecommendationRequest, RecommendationResponse
from schemas.filters import FiltersResponse
import io
import csv
import json
from api.dependencies import get_recommendation_manager
from repositories.user_profile import UserProfileRepository
from managers.recommendation_manager import RecommendationManager

router = APIRouter()

@router.get("/profiles/{user_id}")
async def get_profile(user_id: str, session = Depends(get_recommendation_manager)):
    repo = UserProfileRepository(session)
    profile = await repo.get_by_id(user_id)
    if not profile:
        return {"error": "Not found"}
    return profile

@router.post("/upload-history")
def upload_history(file: UploadFile = File(...)):
    content = file.file.read()
    try:
        if file.filename.endswith(".json"):
            data = json.loads(content)
            movies = [MovieHistoryItem(**m) for m in data["movies"]]
        elif file.filename.endswith(".csv"):
            reader = csv.DictReader(io.StringIO(content.decode()))
            movies = [MovieHistoryItem(**{**row, "genres": row["genres"].split("|"), "countries": row["countries"].split("|")}) for row in reader]
        else:
            raise ValueError("Unsupported file type")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid file: {e}")
    return {"message": f"Parsed {len(movies)} movies"}

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
        countries=["USA", "UK", "FR", "JP"]
    ) 