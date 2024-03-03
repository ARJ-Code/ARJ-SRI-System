import csv
import json
from typing import List
from src.sri.core import Corpus
from src.sri.movie import Movie

class MovieCorpus(Corpus):
    def __init__(self) -> None:
        self.path: str = 'data/movies_db/movies_metadata.csv'
        self.documents: List[Movie] = []

    def load(self) -> List[Movie]:
        def process_genres(genres_str):
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
            try :
                genres = [json.loads(genre) for genre in genres_str]
            except:
                pass
            # we need to extract the name from each dictionary
            return [genre['name'] for genre in genres]

        with open(self.path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.documents.append(Movie(
                    row['title'],
                    row['overview'],
                    process_genres(row['genres']),
                    row['original_language'],
                    row['popularity'],
                    row['vote_average']
                ))
        return self.documents