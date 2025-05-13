import numpy as np

def relevance_expansion(query, relevant_docs, non_relevant_docs, alpha=1.0, beta=0.75, gamma=0.15):
    expanded_query = alpha*query + beta*relevant_docs.mean(axis=0) - gamma*non_relevant_docs.mean(axis=0)
    return np.maximum(expanded_query, 0)


def embedding_expansion(model, headings):
    # Get the embeddings for each node of the tree (headings)
    tree_embeddings = []
    for heading in headings:
        heading_words = heading.split()
        try:
            most_similar = model.wv.most_similar(positive=heading_words, topn=3)
        except KeyError:
            most_similar = []
        enhanced_heading = set(heading_words + [word for word, _ in most_similar])

        # Get the embeddings for the enhanced heading
        heading_embeddings = []
        for word in enhanced_heading:
            try:
                heading_embeddings.append(model.wv[word])
            except KeyError:
                pass
        tree_embeddings.append(np.mean(heading_embeddings, axis=0))

    # Expand the global tree
    embedded_tree = np.mean(tree_embeddings, axis=0)
    cosine_similarities = model.wv.cosine_similarities(embedded_tree, model.wv.vectors)
    most_similar = np.argsort(cosine_similarities)[-3:]
    expanded_tree_embedding = np.mean(embedded_tree + model.wv.vectors[most_similar], axis=0)

    return expanded_tree_embedding