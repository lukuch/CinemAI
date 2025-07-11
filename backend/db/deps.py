from db.engine import AsyncSessionLocal
from contextlib import asynccontextmanager

@asynccontextmanager
async def get_async_session():
    async with AsyncSessionLocal() as session:
        yield session 