from typing import List
from .core import Corpus, Document
import ir_datasets


class IRDataset(Corpus):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def load(self, cant: int = -1) -> List[Document]:
        dataset = ir_datasets.load(self.name)
        self.dataset = dataset

        self.documents = [IRDocument(doc.doc_id, doc.title, doc.text)
                          for doc in dataset.docs_iter()[:cant]]

        return self.documents

    def get_qrels(self) -> List:
        return self.dataset.qrels_iter()

    def get_queries(self) -> List:
        return self.dataset.queries_iter()


class IRDocument(Document):
    def __init__(self, doc_id: str, title: str, text: str) -> None:
        super().__init__(doc_id, title, text)
