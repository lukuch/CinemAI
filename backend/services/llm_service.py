import openai
from domain.interfaces import LLMService
from domain.entities import UserProfile, Movie
from core.settings import settings
from typing import List, Dict, Any

class OpenAILLMService(LLMService):
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-4o"

    def rerank(self, user_profile: UserProfile, candidates: List[Movie]) -> List[Dict[str, Any]]:
        taste_summary = "\n".join([
            f"Cluster {i+1}: {c.count} movies, avg rating {c.average_rating:.2f}"
            for i, c in enumerate(user_profile.clusters)
        ])
        candidate_list = "\n".join([
            f"{i+1}. {m.title} ({m.year}) - {', '.join(m.genres)}"
            for i, m in enumerate(candidates)
        ])
        prompt = f"""
You are an expert movie recommender. Given the following user taste profile and candidate movies, select the top 10 recommendations and provide a brief justification for each.

User Taste Profile:
{taste_summary}

Candidate Movies:
{candidate_list}

For each recommendation, provide:
- Title
- Year
- Genres
- Justification (1-2 sentences)
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7
        )
        return [{"text": response.choices[0].message.content}] 