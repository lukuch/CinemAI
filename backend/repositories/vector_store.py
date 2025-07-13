import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from db.models import UserProfile
from domain.entities import Cluster, Embedding
from domain.entities import UserProfile as DomainUserProfile
from domain.interfaces import VectorStoreRepository
from schemas.watch_history import MovieHistoryItem


class PgvectorRepository(VectorStoreRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_user_profile(self, profile: DomainUserProfile):
        result = await self.session.execute(select(UserProfile).where(UserProfile.user_id == profile.user_id))
        db_profile = result.scalar_one_or_none()
        clusters_json = json.dumps(
            [{"centroid": c.centroid.vector, "average_rating": c.average_rating, "count": c.count} for c in profile.clusters]
        )
        movies_json = json.dumps([m.model_dump() for m in profile.movies] if profile.movies else [])
        if db_profile:
            db_profile.clusters = clusters_json
            db_profile.movies = movies_json
        else:
            db_profile = UserProfile(user_id=profile.user_id, clusters=clusters_json, movies=movies_json)
            self.session.add(db_profile)
        await self.session.commit()
        await self.session.refresh(db_profile)

    async def get_user_profile(self, user_id: str) -> DomainUserProfile:
        result = await self.session.execute(select(UserProfile).where(UserProfile.user_id == user_id))
        db_profile = result.scalar_one_or_none()
        if not db_profile:
            return None
        clusters = [
            Cluster(centroid=Embedding(c["centroid"]), movies=[], average_rating=c["average_rating"], count=c["count"])
            for c in json.loads(db_profile.clusters)
        ]
        movies = []
        if hasattr(db_profile, "movies") and db_profile.movies:
            for m in json.loads(db_profile.movies):
                movies.append(MovieHistoryItem(**m))
        return DomainUserProfile(user_id=user_id, clusters=clusters, movies=movies)
