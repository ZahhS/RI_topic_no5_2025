import numpy as np

def DCG(query_id, documents_ids, qrels):
    dcg = 0.0
    for i, doc_id in enumerate(documents_ids):
        if (query_id, doc_id) in qrels:
            rel = qrels[(query_id, doc_id)]
            dcg += rel / np.log2(i + 2)  # log2(i + 2) car index commence à 0
    return dcg

def NDCG(query_id, documents_ids, qrels):
    ideal_order = sorted(
        documents_ids, 
        key=lambda doc_id: qrels.get((query_id, doc_id), 0), 
        reverse=True
    )
    ideal_dcg = DCG(query_id, ideal_order, qrels)
    if ideal_dcg == 0:
        return 0.0
    return DCG(query_id, documents_ids, qrels) / ideal_dcg


# def AP(query_id, documents_ids, qrels):
#     precision = 0
#     n_relevant_docs_retrieved = 0
#     relevant_docs = [doc_id for (q_id, doc_id) in qrels if query_id == q_id]
#     for i, doc_id in enumerate(documents_ids):
#         if (query_id, doc_id) in qrels:
#             n_relevant_docs_retrieved += 1
#             precision += qrels[(query_id, doc_id)] * n_relevant_docs_retrieved / (i + 1)
#     return precision / len(relevant_docs)

def AP(query_id, documents_ids, qrels):
    relevant_docs = [doc_id for (q_id, doc_id), score in qrels.items() if q_id == query_id]
    if not relevant_docs:
        return 0.0  

    precision = 0.0
    n_relevant_docs_retrieved = 0
    for i, doc_id in enumerate(documents_ids):
        if (query_id, doc_id) in qrels:
            n_relevant_docs_retrieved += 1
            precision += qrels[(query_id, doc_id)] * n_relevant_docs_retrieved / (i + 1)
    return precision / len(relevant_docs)

def RPrec(query_id, documents_ids, qrels):
    all_relevant_docs = [doc_id for (q_id, doc_id) in qrels if q_id == query_id]
    R = len(all_relevant_docs)
    if R == 0:
        return 0.0

    top_R_docs = documents_ids[:R]
    retrieved_relevant = sum(1 for doc_id in top_R_docs if (query_id, doc_id) in qrels)

    return retrieved_relevant / R