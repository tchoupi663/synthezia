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
