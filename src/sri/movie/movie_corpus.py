import csv
import json
from typing import List
from ..core import Corpus
from .movie_doc import Movie


class MovieCorpus(Corpus):
    def __init__(self) -> None:
        '''
        Initializes a MovieCorpus instance.

        Attributes:
        - path: str
            The path to the movies metadata CSV file.
        - documents: List[Movie]
            A list to store Movie objects.
        '''
        self.path: str = 'data/movies_db/movies_metadata.csv'
        self.documents: List[Movie] = []

    def load(self, cant: int = -1) -> List[Movie]:
        '''
        Loads movie data from the CSV file.

        Parameters:
        - cant: int (optional)
            The number of documents to load (-1 loads all).

        Returns:
        - List[Movie]
            A list of Movie objects.
        '''
        def process_genres(genres_str):
            '''
            Processes the genres string and extracts genre names.

            Parameters:
            - genres_str: str
                A string containing genre information.

            Returns:
            - List[str]
                A list of genre names.
            '''
            genres_str = genres_str.replace("'", '"')
            # remove the brackets
            genres_str = genres_str[1:-1]
            # split the string into a list of strings
            genres_str = genres_str.split('}, ')
            # add the closing bracket to the last element
            for i in range(len(genres_str) - 1):
                genres_str[i] = genres_str[i] + '}'
            # we need to convert it to a dictionary
            genres = []
            try:
                genres = [json.loads(genre) for genre in genres_str]
            except:
                pass
            # we need to extract the name from each dictionary
            return [genre['name'] for genre in genres]

        with open(self.path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            ind = 0

            for i, row in enumerate(reader):
                if cant == ind:
                    break

                self.documents.append(Movie(
                    str(i),
                    row['title'],
                    row['overview'],
                    process_genres(row['genres']),
                    row['original_language'],
                    row['popularity'],
                    row['vote_average']
                ))

                ind += 1
        return self.documents
