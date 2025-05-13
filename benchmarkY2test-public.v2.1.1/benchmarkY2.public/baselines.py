from src.expansion import relevance_expansion, embedding_expansion

import src.metrics as metrics
import numpy as np

class RelevanceBaseline(object):
    def __init__(self, dataset, vectorizer):
        self.dataset = dataset
        self.vectorizer = vectorizer
    
    def vectorize(self, text):
        return self.vectorizer.transform([text]).toarray()[0]
    
    def get_query_vector(self, query, expansion=None):
        q = self.vectorize(query[1][1])
        if expansion is not None:
            relevant_docs = self.vectorizer.transform(np.array([doc_id for q_id, doc_id in self.dataset["relevances"].keys() if q_id == query[0]])).toarray()
            non_relevant_docs = self.vectorizer.transform(np.array([doc_id for doc_id in self.dataset["documents"].keys() if doc_id not in relevant_docs])).toarray()
            return relevance_expansion(q, relevant_docs, non_relevant_docs, alpha=expansion[0], beta=expansion[1], gamma=expansion[2])
        else:
            return q
    
    def get_top_k(self, query, k=1000):
        return list(self.dataset["documents"].keys())[:k]

    def eval_query(self, query, k=1000, expansion=None):
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
    
    def eval_model(self, k=1000, expansion=None):
        results = {
            "NDCG": [],
            "MAP": [],
            "RPrec": []
        }
        for query in self.dataset["queries"].items():
            query_results = self.eval_query(query, k, expansion)
            for metric_name, metric_value in query_results.items():
                results[metric_name].append(metric_value)
        results = {metric_name: np.mean(metric_values) for metric_name, metric_values in results.items()}
        return results


class RIH_QL(RelevanceBaseline):
    def __init__(self, dataset, vectorizer):
        self.dataset = dataset
        self.tokenizer = vectorizer.build_tokenizer()
        self.vectorizer = vectorizer

    def get_query_likelihood(self, query, doc):
        d = self.vectorize(doc)
        dl = len(self.tokenizer(doc))
        if dl == 0:
            score = -np.inf
        else:
            score = np.where(query > 0, np.log(d/dl + 1e-12), np.log(1-d/dl + 1e-12)).sum()
        return score

    def get_top_k(self, query, k=1000):
        documents_ids, scores = [], []
        for doc_id, doc in self.dataset["documents"].items():
            documents_ids.append(doc_id)
            scores.append(self.get_query_likelihood(query, doc))
        documents_ids, scores = np.array(documents_ids), np.array(scores)
        top_k_indexes = np.argsort(scores)[::-1][:k]
        return documents_ids[top_k_indexes], scores[top_k_indexes]
    

class RIH_Cosine(RelevanceBaseline):
    def __init__(self, dataset, vectorizer):
        self.dataset = dataset
        self.documents_ids = np.array(list(self.dataset["documents"].keys()))
        self.documents_vectors = vectorizer.transform(list(self.dataset["documents"].values())).toarray()
        self.vectorizer = vectorizer

    def cosine_similarities(self, query):
        dot_product = self.documents_vectors @ query.reshape(-1, 1)
        denominator = (np.maximum(np.linalg.norm(query), 1e-12) * np.maximum(np.linalg.norm(self.documents_vectors), 1e-12))
        return dot_product / denominator

    def get_top_k(self, query, k=1000):
        scores = self.cosine_similarities(query).reshape(-1)
        top_k_indexes = np.argsort(scores)[::-1][:k]
        return self.documents_ids[top_k_indexes], scores[top_k_indexes]
    


class EmbeddingBaseline(object):
    def __init__(self, dataset, w2v_model):
        self.dataset = dataset
        self.documents_ids = np.array(list(self.dataset["documents"].keys()))
        self.w2v_model = w2v_model
        self.embedded_documents = np.array([self.vectorize(doc) for doc in self.dataset["documents"].values()])
        
    def vectorize(self, text):
        embedding = np.zeros(self.w2v_model.vector_size)
        size = 0
        for word in text.split():
            try:
                embedding += self.w2v_model.wv[word]
                size += 1
            except KeyError:
                pass
        if size == 0:
            return embedding
        return embedding / size
    
    def get_query_vector(self, query, expansion=False):
        if expansion:
            return embedding_expansion(self.w2v_model, [query[1][1], *query[1][2]])
        else:
            return self.vectorize(query[1][0])
    
    def cosine_similarities(self, query):
        dot_product = self.embedded_documents @ query.reshape(-1, 1)
        denominator = (np.maximum(np.linalg.norm(query), 1e-12) * np.maximum(np.linalg.norm(self.embedded_documents), 1e-12))
        return dot_product / denominator
    
    def get_top_k(self, query, k=1000):
        scores = self.cosine_similarities(query).reshape(-1)
        top_k_indexes = np.argsort(scores)[::-1][:k]
        return self.documents_ids[top_k_indexes], scores[top_k_indexes]

    def eval_query(self, query, k=1000, expansion=False):
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
    
    def eval_model(self, k=1000, expansion=False):
        results = {
            "NDCG": [],
            "MAP": [],
            "RPrec": []
        }
        for query in self.dataset["queries"].items():
            query_results = self.eval_query(query, k, expansion)
            for metric_name, metric_value in query_results.items():
                results[metric_name].append(metric_value)
        results = {metric_name: np.mean(metric_values) for metric_name, metric_values in results.items()}
        return results