from sqlalchemy import JSON, Column, String

from db.base import Base


class UserProfile(Base):
    __tablename__ = "user_profiles"
    user_id = Column(String, primary_key=True, index=True)
    clusters = Column(JSON, nullable=False)
    movies = Column(JSON)
