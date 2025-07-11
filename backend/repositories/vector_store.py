from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from domain.interfaces import VectorStoreRepository
from domain.entities import UserProfile as DomainUserProfile, Cluster, Embedding
from db.models import UserProfile
import json

class PgvectorRepository(VectorStoreRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_user_profile(self, profile: DomainUserProfile):
        result = await self.session.execute(
            select(UserProfile).where(UserProfile.user_id == profile.user_id)
        )
        db_profile = result.scalar_one_or_none()
        clusters_json = json.dumps([
            {
                "centroid": c.centroid.vector,
                "average_rating": c.average_rating,
                "count": c.count
            } for c in profile.clusters
        ])
        if db_profile:
            db_profile.clusters = clusters_json
        else:
            db_profile = UserProfile(user_id=profile.user_id, clusters=clusters_json)
            self.session.add(db_profile)
        await self.session.commit()
        await self.session.refresh(db_profile)

    async def get_user_profile(self, user_id: str) -> DomainUserProfile:
        result = await self.session.execute(
            select(UserProfile).where(UserProfile.user_id == user_id)
        )
        db_profile = result.scalar_one_or_none()
        if not db_profile:
            return None
        clusters = []
        for c in json.loads(db_profile.clusters):
            clusters.append(Cluster(
                centroid=Embedding(c["centroid"]),
                movies=[],
                average_rating=c["average_rating"],
                count=c["count"]
            ))
        return DomainUserProfile(user_id=user_id, clusters=clusters) 