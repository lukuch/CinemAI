import orjson
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
        clusters_list = [
            {
                "centroid": c.centroid.vector.tolist() if hasattr(c.centroid.vector, "tolist") else c.centroid.vector,
                "average_rating": float(c.average_rating),
                "count": int(c.count),
            }
            for c in profile.clusters
        ]
        movies_list = [m.model_dump() for m in profile.movies] if profile.movies else []
        if db_profile:
            db_profile.clusters = clusters_list
            db_profile.movies = movies_list
        else:
            db_profile = UserProfile(user_id=profile.user_id, clusters=clusters_list, movies=movies_list)
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
            for c in (db_profile.clusters or [])
        ]
        movies = [MovieHistoryItem(**m) for m in (db_profile.movies or [])]
        return DomainUserProfile(user_id=user_id, clusters=clusters, movies=movies)
