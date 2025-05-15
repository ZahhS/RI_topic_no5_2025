import numpy as np

def DCG(query_id, documents_ids, qrels):
    dcg = 0
    for i, doc_id in enumerate(documents_ids):
        if (query_id, doc_id) in qrels:
            dcg += qrels[(query_id, doc_id)] / np.log2(i + 2)
    return dcg

def NDCG(query_id, documents_ids, qrels):
    ideal_order = sorted(documents_ids, key=lambda doc_id: qrels.get((query_id, doc_id), 0), reverse=True)
    ideal_dcg = DCG(query_id, ideal_order, qrels)
    if ideal_dcg == 0:
        return 0
    return DCG(query_id, documents_ids, qrels) / ideal_dcg


def AP(query_id, documents_ids, qrels):
    precision = 0
    n_relevant_docs_retrieved = 0
    relevant_docs = [doc_id for (q_id, doc_id) in qrels if query_id == q_id]
    for i, doc_id in enumerate(documents_ids):
        if (query_id, doc_id) in qrels:
            n_relevant_docs_retrieved += 1
            precision += qrels[(query_id, doc_id)] * n_relevant_docs_retrieved / (i + 1)
    return precision / len(relevant_docs)


def RPrec(query_id, documents_ids, qrels):
    relevant_docs = [doc_id for doc_id in documents_ids if (query_id, doc_id) in qrels]
    all_relevant_docs = [doc_id for doc_id in qrels if doc_id[0] == query_id]
    return len(relevant_docs) / len(all_relevant_docs)