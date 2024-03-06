from sri.models.vectorial import Vectorial
from sri.models.boolean import Boolean
from sri.models.lsi import LSI
import sys
from sri.sri import SRISystem
from sri.ir_dataset import IRDataset
from ir_measures import calc_aggregate
from ir_measures.measures import nDCG, P, RR, AP

cant = -1

try:
    cant = int(sys.argv[1])
except:
    pass

# corpus = IRDataset("beir/arguana")
corpus = IRDataset("cranfield")
corpus.load(cant)

vectorial_model = Vectorial()
lsi_model = LSI()
boolean_model = Boolean()

models = [vectorial_model, lsi_model]
name_models = ['vectorial', 'lsi']

sri = SRISystem(models)
sri.load()


qrels = corpus.get_qrels()
queries = list(corpus.get_queries())


qrels = [q for q in qrels if any(
    [doc for doc in corpus.documents if doc.doc_id == q.doc_id])]


for i in range(len(models)):
    sri.change_selected(i)

    results = {}

    for query in queries:
        top_k_results = sri.query(query.text)
        results[query.query_id] = {doc_id: v for doc_id, _, v in top_k_results}

    # Calcular las métricas
    metrics = calc_aggregate(
        [nDCG@10, P@5, RR(rel=2), AP(rel=2)], qrels, results)

    print(f'Model: {name_models[i]}')
    # Imprimir las métricas
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print('*****************')
