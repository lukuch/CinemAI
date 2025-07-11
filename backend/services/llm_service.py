import openai
from domain.interfaces import LLMService
from domain.entities import UserProfile, Movie
from core.settings import settings
from typing import List, Dict, Any
from pydantic import BaseModel, RootModel
from typing import List as TypingList
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

class Recommendation(BaseModel):
    title: str
    year: int
    genres: TypingList[str]
    justification: str

class Recommendations(RootModel[TypingList[Recommendation]]):
    pass

class OpenAILLMService(LLMService):
    def __init__(self):
        self.model = "gpt-4o"
        self.llm = ChatOpenAI(api_key=settings.openai_api_key, model_name=self.model)
        self.parser = PydanticOutputParser(pydantic_object=Recommendations)

    def rerank(self, user_profile: UserProfile, candidates: List[Movie]) -> List[Dict[str, Any]]:
        taste_summary = "\n".join([
            f"Cluster {i+1}: {c.count} movies, avg rating {c.average_rating:.2f}"
            for i, c in enumerate(user_profile.clusters)
        ])
        candidate_list = "\n".join([
            f"{i+1}. {m.title} ({m.year}) - {', '.join(m.genres)}"
            for i, m in enumerate(candidates)
        ])
        prompt = PromptTemplate(
            template="""
You are an expert movie recommender. Given the following user taste profile and candidate movies, select the top 10 recommendations and provide a brief justification for each.

User Taste Profile:
{taste_summary}

Candidate Movies:
{candidate_list}

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
        result = chain.invoke({
            "taste_summary": taste_summary,
            "candidate_list": candidate_list,
        })
        return [r.dict() for r in result.root] 