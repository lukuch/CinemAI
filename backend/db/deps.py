from typing import AsyncGenerator

from db.engine import AsyncSessionLocal


async def get_async_session() -> AsyncGenerator[AsyncSessionLocal, None]:
    async with AsyncSessionLocal() as session:
        yield session
