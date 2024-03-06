from typing import List
from ..core import Document


class Movie(Document):
    def __init__(self,
                 doc_id: str,
                 title: str,
                 overview: str,
                 genres: List[str],
                 original_language: str,
                 popularity: str,
                 vote_average: str) -> None:
        '''
        Initializes a Movie object.

        Parameters:
        - title: str
            The title of the movie.
        - overview: str
            A brief overview or summary of the movie.
        - genres: List[str]
            A list of genre names associated with the movie.
        - original_language: str
            The original language of the movie.
        - popularity: str
            A measure of the movie's popularity.
        - vote_average: str
            The average rating or score given to the movie.
        '''
        super().__init__(doc_id, title, overview)
        self.genres: str = genres
        self.original_language = original_language
        self.popularity = popularity
        self.vote_average = vote_average
