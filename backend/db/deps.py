from db.engine import AsyncSessionLocal
from typing import AsyncGenerator

async def get_async_session() -> AsyncGenerator[AsyncSessionLocal, None]:
    async with AsyncSessionLocal() as session:
        yield session 