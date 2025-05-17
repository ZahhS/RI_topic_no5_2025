

# class KnowledgeBase(object):
#     def __init__(self):
#         self.kb = set()

#     def build(self, dataset):
#         for query in dataset:
#             # The title of the article is the entity
#             entity = query[2]
#             self.kb.add(entity)

#     def save(self, filename):
#         pickle.dump(self.kb, open(filename, 'wb'))

#     def load(self, filename):
#         self.kb = pickle.load(open(filename, 'rb'))


# if __name__ == '__main__':
#     final_knowledge_base = set()

#     for i in range(5):
#         dataset = ir_datasets.load(f"car/v1.5/train/fold{i}")

#         knowledge_base = KnowledgeBase()
#         knowledge_base.build(dataset.queries_iter())

#         final_knowledge_base = final_knowledge_base | knowledge_base.kb

#     pickle.dump(final_knowledge_base, open("../models/knowledge_base.pkl", 'wb'))
import pickle

class KnowledgeBase(object):
    def __init__(self):
        self.kb = set()

    def build_from_local(self, queries):
        for query_id, query_tuple in queries.items():
            entity = query_tuple[1]  # La racine = titre de l’article
            self.kb.add(entity)

    def save(self, filename):
        pickle.dump(self.kb, open(filename, 'wb'))

    def load(self, filename):
        self.kb = pickle.load(open(filename, 'rb'))


if __name__ == '__main__':
    queries = pickle.load(open("queries.pkl", "rb"))

    knowledge_base = KnowledgeBase()
    knowledge_base.build_from_local(queries)

    # Optionnel : vérifier quelques entités
    print("Exemples d'entités dans la KB :")
    print(list(knowledge_base.kb)[:10])

    # Sauvegarde
    knowledge_base.save("knowledge_base.pkl")
