from collections import Counter
from typing import Any, Dict
from typing import List
from typing import List as TypingList

from injector import inject
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, RootModel
from structlog.stdlib import BoundLogger

from core.settings import settings
from domain.entities import Movie, UserProfile
from domain.interfaces import LLMService


class Recommendation(BaseModel):
    title: str
    year: int
    genres: TypingList[str]
    justification: str


class Recommendations(RootModel[TypingList[Recommendation]]):
    pass


class OpenAILLMService(LLMService):
    @inject
    def __init__(self, logger: BoundLogger):
        self.model = "gpt-4o"
        self.llm = ChatOpenAI(api_key=settings.openai_api_key, model_name=self.model)
        self.parser = PydanticOutputParser(pydantic_object=Recommendations)
        self.logger = logger

    def rerank(self, user_profile: UserProfile, candidates: List[Movie]) -> List[Dict[str, Any]]:
        self.logger.info("Starting LLM reranking", candidates_count=len(candidates), clusters_count=len(user_profile.clusters))
        taste_summary = self._build_taste_summary(user_profile)
        candidate_list = "\n".join([f"{i+1}. {m.title} ({m.year}) - {', '.join(m.genres)}" for i, m in enumerate(candidates)])
        prompt = PromptTemplate(
            template="""
You are an expert movie recommender. Given the following user taste profile and candidate movies, select the top 10 recommendations and provide a brief justification for each.

User Taste Profile:
{taste_summary}

Candidate Movies:
{candidate_list}

When writing your justifications, DO NOT mention 'taste group', 'cluster', or any internal group numbers. Instead, explain your reasoning in terms of genres, countries, and the user's preferences.

Return your answer as a JSON array, where each item has:
- title (string)
- year (integer)
- genres (list of strings)
- justification (string, 1-2 sentences)

{format_instructions}
""",
            input_variables=["taste_summary", "candidate_list"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        chain = prompt | self.llm | self.parser
        result = chain.invoke(
            {
                "taste_summary": taste_summary,
                "candidate_list": candidate_list,
            }
        )
        reranked_results = [r.dict() for r in result.root]
        self.logger.info(
            "LLM reranking completed",
            results_count=len(reranked_results),
            taste_summary_length=len(taste_summary),
            candidate_list_length=len(candidate_list),
        )
        return reranked_results

    def _build_taste_summary(self, user_profile: UserProfile) -> str:
        def summarize_top(counter, label):
            if not counter:
                return f"no {label} found"
            top = [g for g, _ in counter.most_common(2)]
            return ", ".join(top) if top else f"varied {label}"

        def summarize_cluster(cluster, idx):
            all_genres = [genre for movie in cluster.movies for genre in getattr(movie, "genres", [])]
            all_countries = [country for movie in cluster.movies for country in getattr(movie, "countries", [])]
            genre_counts = Counter(all_genres)
            country_counts = Counter(all_countries)
            top_genres = summarize_top(genre_counts, "genres")
            top_countries = summarize_top(country_counts, "countries")
            return (
                f"Taste group {idx+1}: {cluster.count} movies, avg rating {cluster.average_rating:.2f}, "
                f"main genres: {top_genres}, main countries: {top_countries}"
            )

        summary = "\n".join(summarize_cluster(cluster, i) for i, cluster in enumerate(user_profile.clusters))

        self.logger.info(
            "Taste summary built",
            clusters_count=len(user_profile.clusters),
            summary_length=len(summary),
            total_movies=sum(len(cluster.movies) for cluster in user_profile.clusters),
        )

        return summary
