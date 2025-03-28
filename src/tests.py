from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import json

# import nltk
# nltk.download('punkt')


def extract_text(data):
    combined_text = f"""
    Context: {', '.join(data['éléments_contexte'])}.
    Decision: {data['décision']}.
    Summary: {data['résumé']}.
    Articles: {', '.join(data['articles_pertinents'])}.
    """
    return combined_text


def extract_metadata(data):
    metadata = {
        "lien_internet": data["lien_internet"],
        "identifiant_document": data["identifiant_document"],
        "date": data["date"],
        "décision": data["décision"],
    }
    return metadata


cases = [
    {
        "lien_internet": "https://www.legifrance.gouv.fr/juri/id/JURITEXT000051243781?page=1&pageSize=100&searchField=ALL&searchType=ALL&sortValue=DATE_DESC&tab_selection=juri&typePagination=DEFAULT",
        "identifiant_document": "D 23-23.693",
        "date": "2025-02-12",
        "décision": "cassation partielle cour appel versailles",
        "éléments_contexte": [
            "licenciement",
            "heures supplémentaires",
            "travail dissimulé",
        ],
        "résumé": "salariée licenciée pour faute grave saisit prud'hommes demande paiement heures supplémentaires indemnités cour cassation casse arrêt cour appel partiellement déboute salariée demande heures supplémentaires dommages intérêts absence repos hebdomadaire indemnité travail dissimulé",
        "articles_pertinents": [
            "Article 562 du code de procédure civile",
            "Article L. 3171-4 du code du travail",
        ],
    },
    {
        "lien_internet": "https://www.legifrance.gouv.fr/juri/id/JURITEXT000051311772?page=1&pageSize=100&searchField=ALL&searchType=ALL&sortValue=DATE_DESC&tab_selection=juri&typePagination=DEFAULT",
        "identifiant_document": "V 23-23.340",
        "date": "2025-03-05",
        "décision": "cassation partielle cour appel versailles",
        "éléments_contexte": [
            "droit travail",
            "heures supplémentaires",
            "harcèlement moral",
            "rémunération variable",
            "licenciement",
        ],
        "résumé": "salarié licencié conteste licenciement demande paiement heures supplémentaires dommages intérêts harcèlement moral rémunération variable cour cassation casse arrêt cour appel versailles rejette demandes salarié",
        "articles_pertinents": [
            "Article L. 3111-2 du code du travail",
            "Article L. 1152-1 du code du travail",
            "Article L. 1154-1 du code du travail",
            "Article 16 du code de procédure civile",
            "Article L. 1152-4 du code du travail",
        ],
    },
    {
        "lien_internet": "https://www.legifrance.gouv.fr/juri/id/JURITEXT000051243836?page=1&pageSize=100&searchField=ALL&searchType=ALL&sortValue=DATE_DESC&tab_selection=juri&typePagination=DEFAULT",
        "identifiant_document": "Z 23-17.755",
        "date": "2025-02-13",
        "décision": "cassation partielle rejet demande acquéreur concernant travaux parachèvement",
        "éléments_contexte": [
            "vente état futur achèvement",
            "pénalités retard",
            "vices apparents",
        ],
        "résumé": "société francelot vend immeuble m e délai livraison dépassé acquéreur assigne fixation date achèvement paiement indemnités cour cassation casse arrêt cour appel rejette demande acquéreur concernant travaux parachèvement",
        "articles_pertinents": [
            "Article R. 261-1 du code de la construction et de l'habitation",
            "Article 1642-1 du code civil",
            "Article 1648 du code civil",
        ],
    },
    {
        "lien_internet": "https://www.legifrance.gouv.fr/juri/id/JURITEXT000051151386?page=1&pageSize=100&searchField=ALL&searchType=ALL&sortValue=DATE_DESC&tab_selection=juri&typePagination=DEFAULT",
        "identifiant_document": "H 23-12.932",
        "date": "2025-01-30",
        "décision": "annulation arrêt cour appel renvoi cour appel bordeaux",
        "éléments_contexte": [
            "sécurité sociale",
            "maladie professionnelle",
            "secret médical",
            "audiogramme",
        ],
        "résumé": "caisse primaire assurance maladie landes pourvoi contre arrêt cour appel pau concernant prise charge maladie professionnelle salarié cour cassation annule arrêt cour appel motif secret médical audiogramme",
        "articles_pertinents": [
            "L. 1110-4 du code de la santé publique",
            "L. 315-1 V du code de la sécurité sociale",
            "L. 461-1 du code de la sécurité sociale",
            "R. 441-13 du code de la sécurité sociale",
            "R. 441-14 du code de la sécurité sociale",
            "tableau n° 42 des maladies professionnelles",
        ],
    },
]


texts = [extract_text(case) for case in cases]
# metadata_list = [extract_metadata(case) for case in cases]

stemmer = SnowballStemmer("french")


def stemmed_words(text):
    tokens = word_tokenize(text)
    return [stemmer.stem(word) for word in tokens if word.isalpha()]


vectorizer = TfidfVectorizer(
    strip_accents="ascii",
    lowercase=True,
    norm="l2",
    tokenizer=stemmed_words,
    token_pattern=None,
)

tfidf_matrix = vectorizer.fit_transform(texts)

cosine_sim_matrix = cosine_similarity(tfidf_matrix)

for i in range(len(texts)):
    for j in range(len(texts)):
        if i == j:
            continue
        similarity = cosine_sim_matrix[i][j] * 100
        print(
            f"Similarity between document {i+1} and document {j+1}: {similarity:.2f}%"
        )
        # print(f"  Document {i+1} Metadata: {metadata_list[i]}")
        # print(f"  Document {j+1} Metadata: {metadata_list[j]}")
        print("---")
