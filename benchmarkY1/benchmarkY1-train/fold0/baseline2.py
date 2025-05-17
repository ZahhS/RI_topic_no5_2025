
from expansion2 import get_leafs_name_expansion,get_alias_expansion
import metrics as metrics
import numpy as np
from baselines import EmbeddingBaseline

class RIH_IDs_Embedding(Object):
     def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

    def get_query_vector(self, query):
        headings = [query[1][1]] + list(query[1][2])
        
        entity_ids = [h.replace(" ", "_") + ".id" for h in headings]
        
        vectors = []
        for eid in entity_ids:
            if eid in self.model.wv:
                vectors.append(self.model.wv[eid])
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.model.vector_size)

    def cosine_similarities(self, query_vec):
        dot_products = self.document_embeddings @ query_vec.reshape(-1, 1)
        norm_query = np.linalg.norm(query_vec)
        norms_docs = np.linalg.norm(self.document_embeddings, axis=1)
        cosine_scores = dot_products.flatten() / (norm_query * norms_docs + 1e-12)
        return cosine_scores

    def get_top_k(self, query, k=1000):
        q_vec = self.get_query_vector(query)
        scores = self.cosine_similarities(q_vec)
        top_k_idx = np.argsort(scores)[::-1][:k]
        top_k_docs = [self.dataset["documents_ids"][i] for i in top_k_idx]
        top_k_scores = scores[top_k_idx]
        return top_k_docs, top_k_scores

    def eval_query(self, query, k=100,expansion=False):
        scores = {
        "NDCG": metrics.NDCG,
        "MAP": metrics.AP,
        "RPrec": metrics.RPrec
        }
        results = {}
        q = self.get_query_vector(query, expansion)
        docs, _ = self.get_top_k(q, k)
        for metric_name, metric_callback in scores.items():
            results[metric_name] = metric_callback(query[0], docs, self.dataset["relevances"])
        return results

    def eval_model(self, k=1000):

        results = {
            "NDCG": [],
            "MAP": [],
            "RPrec": []
        }
        for query in self.dataset["queries"].items():
            query_results = self.eval_query(query, k)
            for metric_name, metric_value in query_results.items():
                results[metric_name].append(metric_value)
        results = {metric: np.mean(values) for metric, values in results.items()}
        return results



class EmbeddingBaselineHNames(EmbeddingBaseline):
    def get_query_vector(self, query, expansion=False):
        if expansion:
            leaf_heading=query[1][2][-1] if query[1][2] else query[1][1] #le premier si y'a RIH le deuxième si y'a R-H
            return get_leafs_name_expansion(self.w2v_model, leaf_heading)
        else:
            return self.vectorize(query[1][0])

class EmbeddingBaselineAliases(EmbeddingBaseline):
    def get_query_vector(self, query, expansion=False):
        if expansion:
            leaf_heading=query[1][0]  #On prends le root
            return get_alias_expansion(self.w2v_model, leaf_heading)
        else:
            return self.vectorize(query[1][0])