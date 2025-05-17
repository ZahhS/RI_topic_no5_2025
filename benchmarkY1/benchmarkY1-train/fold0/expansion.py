import numpy as np

def relevance_expansion(query, relevant_docs, non_relevant_docs, alpha=1.0, beta=0.75, gamma=0.15):
    expanded_query = alpha*query + beta*relevant_docs.mean(axis=0) - gamma*non_relevant_docs.mean(axis=0)
    return np.maximum(expanded_query, 0)


def embedding_expansion(model, headings):
    tree_embeddings = []

    for heading in headings:
        heading_words = heading.split()
        try:
            most_similar = [word for word, _ in model.wv.most_similar(positive=heading_words, topn=3)]
        except KeyError:
            most_similar = []

        enhanced_heading = set(heading_words + most_similar)
        heading_vectors = []

        for word in enhanced_heading:
            if word in model.wv:
                heading_vectors.append(model.wv[word])

        if heading_vectors:  # ajouter uniquement si non vide
            heading_embedding = np.mean(heading_vectors, axis=0)
            tree_embeddings.append(heading_embedding)

    # Si aucune embedding valide n’a été ajoutée
    if len(tree_embeddings) == 0:
        print("Aucune entité trouvée pour cette requête. Expansion désactivée.")
        return np.zeros(model.vector_size)

    try:
        embedded_tree = np.mean(np.vstack(tree_embeddings), axis=0)
    except Exception as e:
        print("Erreur lors du calcul de embedded_tree :", e)
        return np.zeros(model.vector_size)

    try:
        cosine_similarities = model.wv.cosine_similarities(embedded_tree, model.wv.vectors)
        most_similar = np.argsort(cosine_similarities)[-3:]
        expanded_tree_embedding = np.mean(
            np.vstack([embedded_tree, model.wv.vectors[most_similar]]), axis=0
        )
        return expanded_tree_embedding
    except Exception as e:
        print("Erreur lors de l’expansion finale :", e)
        return embedded_tree  # fallback : utiliser embedded_tree seul