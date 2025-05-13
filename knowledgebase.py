import dill as pickle
import ir_datasets

class KnowledgeBase(object):
    def __init__(self):
        self.kb = set()

    def build(self, dataset):
        for query in dataset:
            # The title of the article is the entity
            entity = query[2]
            self.kb.add(entity)

    def save(self, filename):
        pickle.dump(self.kb, open(filename, 'wb'))

    def load(self, filename):
        self.kb = pickle.load(open(filename, 'rb'))


if __name__ == '__main__':
    final_knowledge_base = set()

    for i in range(5):
        dataset = ir_datasets.load(f"car/v1.5/train/fold{i}")

        knowledge_base = KnowledgeBase()
        knowledge_base.build(dataset.queries_iter())

        final_knowledge_base = final_knowledge_base | knowledge_base.kb

    pickle.dump(final_knowledge_base, open("../models/knowledge_base.pkl", 'wb'))