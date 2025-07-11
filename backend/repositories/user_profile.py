from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from db.models import UserProfile

class UserProfileRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, user_id: str):
        result = await self.session.execute(
            select(UserProfile).where(UserProfile.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def create(self, user_id: str, clusters: dict):
        profile = UserProfile(user_id=user_id, clusters=clusters)
        self.session.add(profile)
        await self.session.commit()
        await self.session.refresh(profile)
        return profile 