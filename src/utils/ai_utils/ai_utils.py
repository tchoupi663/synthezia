from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import utils.file_utils.file_utils as fu
import utils.terminal_utils.terminal_utils as tu
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import utils.file_utils.file_utils as fu
import re
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json


class DocumentJuridique(BaseModel):
    lien_internet: str
    identifiant_document: str
    date: str
    décision: str
    éléments_contexte: list[str]
    résumé: str
    articles_pertinents: list[str]


def extraction_donnees_importantes(content: str):
    load_dotenv()
    api_key = os.getenv("API_KEY")
    client = genai.Client(api_key=api_key)

    generate_content_config = types.GenerateContentConfig(
        temperature=0.2,
        top_p=0.95,
        top_k=40,
        max_output_tokens=1024,
        response_mime_type="application/json",
        response_schema=list[DocumentJuridique],
        system_instruction="""
Créez un fichier JSON structuré à partir des informations suivantes extraites d'un document de jurisprudence.
Assurez-vous que le résumé est clair et concis, tout en conservant les informations pertinentes pour faciliter la recherche et la comparaison avec d'autres documents. 
Cela permettra de proposer à l'utilisateur les documents les plus pertinents à sa situation à partir d'un résumé de la situation écrite par l'utilisateur. 
Tu devras également inclure des éléments de contexte et des articles pertinents pour chaque document.
Effectues la tache en prenant en compte que plus tard tu devras classer les documents en fonction de leur pertinence pour l'utilisateur.
POUR LE RÉSUME, LA DÉCISION ET LES ÉLÉMENTS DE CONTEXTE, NE PAS ECRIRE DE CONNECTEURS LOGIQUES (et, ou, mais, donc, car, ni, or, etc.), NE PAS ECRIRE EN LETTRE CAPITALES, NE PAS METTRE DE PONCTUATIONS.

Exemple de la manière dont les données pourraient apparaître en utilisant le schéma
document_json = {
    "lien_internet": "https://www.legifrance.gouv.fr/juri/id/JURITEXT000051311794?page=1&pageSize=100&searchField=ALL&searchType=ALL&sortValue=DATE_DESC&tab_selection=juri&typePagination=DEFAULT",
    "identifiant_document": "K 22-20.935",
    "date": "2023-01-15",
    "décision": "tribunal statué faveur demandeur",
    "éléments_contexte": ["droit travail", "licenciement abusif", "indemnisation"],
    "résumé": "tribunal jugé défendeur licencié abusivement demandeur ordonné indemnisation",
    "articles_pertinents": ["Article L1234-5 du Code du travail"]
}""",
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=content,
        config=generate_content_config,
    )

    if response:
        fu.save_response_as_json(response.text)


def process_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()
            extraction_donnees_importantes(content)
    except UnicodeDecodeError:
        pass


def traitement_textes():
    files = os.listdir("./donnees")
    text_files = [os.path.join("./donnees", file) for file in files]

    total_items = len(text_files)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(process_file, text_file): text_file
            for text_file in text_files
        }

        for i, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as e:
                print(f"Erreur lors du traitement du fichier {futures[future]} : {e}")
            tu.loading_bar(i, total_items)

    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()
    tu.print_green("Extraction des données réussie.")


nlp = spacy.load("fr_core_news_sm")


def extract_text_and_preprocess(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            cases = json.load(file)

        normalized_texts = []
        metadata = []
        for idx, case in enumerate(cases):
            text = f"SUMMARY: {case['résumé']} CONTEXT: {' '.join(case['éléments_contexte'])} ARTICLES: {' '.join(case['articles_pertinents'])}"
            text = text.lower()
            text = re.sub(r"[^\w\s]", "", text)
            doc = nlp(text)
            tokens = [
                token.lemma_
                for token in doc
                if token.text not in stopwords.words("french") and token.is_alpha
            ]
            normalized_text = " ".join(tokens)
            normalized_texts.append(normalized_text)
            metadata.append(
                {
                    "file": os.path.basename(file_path),
                    "case_index": idx,
                    "identifiant_document": case.get("identifiant_document", "N/A"),
                    "date": case.get("date", "N/A"),
                    "décision": case.get("décision", "N/A"),
                    "summary": case["résumé"],
                    "context": case["éléments_contexte"],
                    "articles": case["articles_pertinents"],
                }
            )

        return normalized_texts, metadata
    except (KeyError, Exception, json.JSONDecodeError) as e:
        print(f"Erreur -> {file_path}: {e}")
        return [], []


def normalize_user_input(user_input):
    user_input = user_input.lower()
    user_input = re.sub(r"[^\w\s]", "", user_input)
    doc = nlp(user_input)
    tokens = [
        token.lemma_
        for token in doc
        if token.text not in stopwords.words("french") and token.is_alpha
    ]
    return " ".join(tokens)


def show_results(similarity_list, metadata_list):
    print("Similarity results with user's description:")
    for i, similarity in similarity_list:
        meta_i = metadata_list[i]
        if similarity < 1:
            continue
        print(
            f"Similarity between user's description and document {i+1} (File: {meta_i['file']}, Case: {meta_i['case_index']}, ID: {meta_i['identifiant_document']}): {similarity:.2f}%"
        )
        print(f"Summary of Document {i+1}: {meta_i['summary']}")
        print(f"Context of Document {i+1}: {', '.join(meta_i['context'])}")
        print(f"Articles of Document {i+1}: {', '.join(meta_i['articles'])}")
        print("---")


def find_recommandations(usr_input: str):
    results = []
    metadata_list = []

    json_folder = "./donnees_json"

    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

    for json_file in json_files:
        file_path = os.path.join(json_folder, json_file)
        texts, metadata = extract_text_and_preprocess(file_path)
        if texts:
            results.extend(texts)
            metadata_list.extend(metadata)

    user_description = usr_input

    normalized_user_description = normalize_user_input(user_description)

    results.append(normalized_user_description)

    vectorizer = TfidfVectorizer(
        strip_accents="ascii",
        lowercase=True,
        norm="l2",
    )

    tfidf_matrix = vectorizer.fit_transform(results)

    cosine_sim_matrix = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    similarity_list = [
        (i, similarity * 100) for i, similarity in enumerate(cosine_sim_matrix[0])
    ]

    similarity_list.sort(key=lambda x: x[1], reverse=True)
    show_results(similarity_list, metadata_list)
