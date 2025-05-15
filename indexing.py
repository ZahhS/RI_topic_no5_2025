from nltk import download, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

import ir_datasets
import joblib
import re


def preprocess(text):
        text = re.sub(r"[^a-zA-Z]", " ", text.lower())
        words = word_tokenize(text)
        words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in words if w not in english_stopwords]
        return words_lemmed

def generate_docs_texts():
        for doc in dataset.docs_iter():
            yield doc[1]


if __name__ == "__main__":
    # Ressources Installation for NLTK
    # download("punkt")
    # download("stopwords")
    # download("wordnet")
    english_stopwords = stopwords.words("english")

    # Load the dataset "TREC CAR 2017" from ir_datasets
    dataset = ir_datasets.load("car/v1.5")
    # Create the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(tokenizer=preprocess, stop_words=english_stopwords)
    # Train the TFIDF-Vectorizer on the corpus
    vectorizer = vectorizer.fit(generate_docs_texts())
    # Save the vectorizer for future use
    joblib.dump(vectorizer, "../models/vectorizer.pkl")


    index, inverted_index = {}, {}

    tokenizer = vectorizer.build_tokenizer()
    for doc in dataset.docs_iter():
        d = vectorizer.transform([doc[1]]).toarray()[0]
        dl = len(tokenizer(doc[1]))
        index[doc[0]] = {"n_tokens": dl, "vector": d}
        
        for i, term in enumerate(vectorizer.get_feature_names_out()):
            inverted_index[term] = inverted_index.get(term, []) + [doc[0]]

    joblib.dump(index, "../indexes/index.pkl")
    joblib.dump(inverted_index, "../indexes/inverted_index.pkl")