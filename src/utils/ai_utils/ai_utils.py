from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import utils.file_utils.file_utils as fu
import utils.terminal_utils.terminal_utils as tu


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
        max_output_tokens=2048,
        response_mime_type="application/json",
        response_schema=list[DocumentJuridique],
        system_instruction="""
Créez un fichier JSON structuré à partir des informations suivantes extraites d'un document de jurisprudence.
Assurez-vous que le résumé est clair et concis, tout en conservant les informations pertinentes pour faciliter la recherche et la comparaison avec d'autres documents. 
Cela permettra de proposer à l'utilisateur les documents les plus pertinents à sa situation à partir d'un résumé de la situation écrite par l'utilisateur. 
Tu devras également inclure des éléments de contexte et des articles pertinents pour chaque document.
Effectues la tache en prenant en compte que plus tard tu devras classer les documents en fonction de leur pertinence pour l'utilisateur.

Exemple de la manière dont les données pourraient apparaître en utilisant le schéma
document_json = {
    "lien_internet": "https://www.legifrance.gouv.fr/juri/id/JURITEXT000051311794?page=1&pageSize=100&searchField=ALL&searchType=ALL&sortValue=DATE_DESC&tab_selection=juri&typePagination=DEFAULT",
    "identifiant_document": "K 22-20.935",
    "date": "2023-01-15",
    "décision": "Le tribunal a statué en faveur du demandeur.",
    "éléments_contexte": ["droit du travail", "licenciement abusif", "indemnisation"],
    "résumé": "Le tribunal a jugé que le défendeur avait licencié abusivement le demandeur et a ordonné une indemnisation.",
    "articles_pertinents": ["Article L1234-5 du Code du travail"]
}""",
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=content,
        config=generate_content_config,
    )

    if response:
        # tu.print_green("Extraction des données réussie.")
        # tu.print_green("Sauvegarde des données extraites...")
        fu.save_response_as_json(response.text)


def traitement_dossier_textes():
    files = os.listdir("./donnees")
    text_files = [file for file in files]

    total_items = len(text_files)
    i = 1

    for text_file in text_files:
        file_path = os.path.join("./donnees", text_file)
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            extraction_donnees_importantes(content)
            tu.loading_bar(
                i,
                total_items,
                length=50,
                print_end="\n",
            )
            i += 1
    tu.print_green("Extraction des données réussie.")
