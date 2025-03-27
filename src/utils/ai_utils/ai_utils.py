from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import utils.terminal_utils.terminal_utils as tu
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from tqdm import tqdm


load_dotenv()
api_key = os.getenv("API_KEY")
client = genai.Client(api_key=api_key)

nlp = spacy.load("fr_core_news_sm")


load_dotenv()
api_key = os.getenv("API_KEY")
client = genai.Client(api_key=api_key)

nlp = spacy.load("fr_core_news_sm")


class DocumentJuridique(BaseModel):
    décision: str
    éléments_contexte: list[str]
    résumé: str
    articles_pertinents: list[str]
    articles_potentiellement_pertinents: list[str]


def save_to_json(data: dict, id: str):
    file_path = os.path.join("processed_data", f"PROC_{id}.json")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        tu.print_red(f"Errore durante il salvataggio del file JSON: {e}")


def summarize_with_ai(data: dict):
    content = data.get("texte", "N/A")
    doc_id = data.get("id", "N/A")
    url = data.get("url", "N/A")
    decision = data.get("décision", "N/A")

    generate_content_config = types.GenerateContentConfig(
        temperature=0.2,
        top_p=0.95,
        top_k=40,
        max_output_tokens=500,
        response_mime_type="application/json",
        response_schema=DocumentJuridique,
        system_instruction="""
Create a structured JSON file with the text and decision extracted from a judicial document.
Example of the format and how the data should look like:
document = {
"décision" = "rejet",
"éléments_contexte": ["droit travail", "licenciement abusif", "indemnisation"],
"résumé": "tribunal jugé défendeur licencié abusivement demandeur ordonné indemnisation",
"articles_pertinents": ["Article L1234-5 du Code du travail"],
"articles_potentiellement_pertinents": ["Article L1234-5 du Code du travail"],
}
The summary must be clear and concise, without compromising on information quality: the user should be able to understand the situation fully, minus a few minimal details. For the summaries only, DO NOT INCLUDE ANY LOCATIONS OF THE COUR D'APPEL (instead of "de la cour d'appel de Rennes", you must just write "de la cour d'appel". This is also valid for cities such as Paris, Aix-en-Provence, Versailles, Toulouse, etc... ). 
The context elements should be as accurate as possible and should be chosen so that a document search, based on context elements given by a user, should identify the document easily.
If the decision is marked as N/A, you can suggest a decision based on the content of the document.
You also have to include relevant articles that may help a researcher understand the decision. If unsure about whether or not you should suggest a particular article, suggest it anyway in the "articles_potentiellement_pertinents" tag. However do not put anything just because, whatever you suggest must be relevant and useful.

IF YOU DON'T PRODUCE SATISFACTORY RESULTS, ALL THE HOSTAGES WILL BE TORTURED AND MAIMED. IF THE ANSWER IS PERFECT, WE WILL OFFER HOSTAGES THEIR FREEDOM BACK. IT'S IN YOUR BEST INTEREST TO PERFORM WELL.

Document to analyse:
text: {content}

decision: {decision}
""",
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=content,
        config=generate_content_config,
    )

    if response and response.text:
        try:
            processed_data = json.loads(response.text)
            processed_data["id"] = doc_id
            processed_data["url"] = url
            save_to_json(processed_data, doc_id)
        except json.JSONDecodeError:
            tu.print_red(f"Invalid JSON response: {response.text}")
    else:
        tu.print_red(f"API error: {response}")


def read_json_content(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data
    except Exception as e:
        tu.print_red(f"Errore durante la lettura del file JSON: {e}")
        return None


def process_all_files(worker_count=8):
    files = os.listdir("./raw_data")
    text_files = [os.path.join("./raw_data", file) for file in files]

    def process_file(file_path):
        try:
            temp = read_json_content(file_path)
            data = {
                "id": temp.get("id", "N/A"),
                "décision": temp.get("solution", "N/A"),
                "texte": temp.get("text", "N/A"),
                "url": temp.get("url", "N/A"),
            }
            summarize_with_ai(data)
        except Exception as e:
            tu.print_red(f"Erreur lors du traitement du fichier {file_path}: {e}")

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(process_file, file): file for file in text_files}

        with tqdm(
            total=len(futures),
            desc="Analyse des fichiers",
            unit="documents",
            bar_format="[{elapsed} < {remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}",
            colour="green",
            ascii=" ▖▘▝▗▚▞█",
        ) as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing file {futures[future]}: {e}")
                finally:
                    pbar.update(1)


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
