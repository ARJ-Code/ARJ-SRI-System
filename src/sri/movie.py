from typing import List
from .core import Document

class Movie(Document):
    def __init__(self, 
                 title: str, 
                 overview: str, 
                 genres: List[str], 
                 original_language: str,
                 popularity: str,
                 vote_average: str) -> None:
        super().__init__(title, overview)
        self.genres: str = genres
        self.original_language = original_language
        self.popularity = popularity
        self.vote_average = vote_average