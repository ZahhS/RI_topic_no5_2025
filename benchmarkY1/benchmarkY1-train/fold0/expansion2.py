import numpy as np

def get_rih_id_expansion(model,headings):
    ids = [h.replace(" ", "_") + ".id" for h in headings] # on suppose que tous les mots de la requete sont des entitées c'est pourquoi on rajoute id
    vectors = []
    for eid in ids:
        if eid in model.wv: # si c'est vraiment des entites alors on l'ajoute au vecteur
            vectors.append(model.wv[eid])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

    
def get_leafs_name_expansion(model,heading):
    heading_words = heading.split()
    try:
        most_similar = [word for word, _ in model.wv.most_similar(positive=heading_words, topn=5)]
    except KeyError:
        most_similar = []

    expansion_words = set(heading_words + s)
    
    vectors = []
    for word in expansion_words:
        if word in model.wv:
            vectors.append(model.wv[word])

    #on vérifie qu'il y a bien au moins 1 mot 
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


def get_alias_expansion(model, root_heading):
    root_words = root_heading.split()
    
    try:
        most_similar = [word for word, _ in model.wv.most_similar(positive=root_words, topn=5)]
    except KeyError:
        most_similar = []

    expansion_words = set(root_words + most_similar)

    vectors = []
    for word in expansion_words:
        if word in model.wv:
            vectors.append(model.wv[word])
    
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)