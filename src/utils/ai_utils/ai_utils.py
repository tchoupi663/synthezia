from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from pydantic import BaseModel
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
import utils.terminal_utils.terminal_utils as cp
import string
import pickle


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

nlp = spacy.load("fr_core_news_sm")


class DocumentJuridique(BaseModel):
    décision: str
    éléments_contexte: list[str]
    résumé: str
    articles_pertinents: list[str]
    articles_potentiellement_pertinents: list[str]


def save_to_json(data: dict, id: str):
    file_path = os.path.join("temp_data", f"PROC_{id}.json")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        cp.print_red(f"Errore durante il salvataggio del file JSON: {e}")


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
Create a structured JSON file with the text extracted from a judicial document.  

The JSON file should follow the format of the example below. The keys should be in French, and the values should also be in French.  

Example of the format:  
document = {  
"décision": "rejet",  
"éléments_contexte": ["droit travail", "licenciement abusif", "indemnisation"],  
"résumé": "Le tribunal a jugé que le défendeur a licencié abusivement le demandeur et a ordonné une indemnisation.",  
"articles_pertinents": ["Article L1234-5 du Code du travail"],  
"articles_potentiellement_pertinents": ["Article L1234-5 du Code du travail"]  
}  

### Instructions for the summary:  
- The summary must be clear and concise but must not omit critical details. It should allow the user to fully understand the case, except for minor details.  
- It must include all key financial amounts and legal justifications.  
- It must explicitly mention contradictions or legal inconsistencies cited in the document.  
- It must describe the nature of the investment, the obligations of each party, and the procedural decisions taken.  

### Instructions for categorizing the decision:  
- If the decision is not explicitly stated in the document, you must determine whether it falls under "arrêt," "cassation partielle," or "cassation totale":  
  - "Cassation totale" if the entire reasoning of the appellate court is invalidated.  
  - "Cassation partielle" if only specific provisions are overturned.  
  - "Arrêt" if the decision is upheld.  

### Instructions for legal articles:  
- All legal articles explicitly mentioned in the document must be included in "articles_pertinents."  
- Additional relevant articles that are not cited but could support the decision must be included in "articles_potentiellement_pertinents."  

Ensure that the JSON file is well-structured and contains accurate legal references to facilitate document retrieval based on context elements.

---

decision: {decision}
text: {content}
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
            pass
    else:
        pass


def read_json_content(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            data = json.load(file)
            return data
    except (Exception, json.JSONDecodeError) as e:
        return None


def calculate_success_rate():
    raw_files = 0
    proc_files = 0

    with tqdm(
        total=20000,
        desc="Counting raw files",
        unit="documents",
        bar_format="[{elapsed} < {remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}",
        colour="blue",
        ascii=" ▖▘▝▗▚▞█",
    ) as pbar:
        for file in os.listdir("./raw_data"):
            data = read_json_content(f"raw_data/{file}")
            if data:
                raw_files += 1
            pbar.update(1)

        pbar.write(f"Raw files: {raw_files}")
        pbar.n = 10000
        pbar.set_description("Counting processed files")

        for file in os.listdir("./temp_data"):
            data = read_json_content(f"temp_data/{file}")
            if data:
                proc_files += 1
            pbar.update(1)

        pbar.write(f"Processed files: {proc_files}")
        pbar.n = 20000

    success_rate = (proc_files / raw_files) * 100

    if success_rate < 80:
        cp.print_red(
            f"File summarization success rate: {success_rate:.2f}% ({proc_files}/{raw_files})"
        )
    else:
        cp.print_green(
            f"File summarization success rate: {success_rate:.2f}% ({proc_files}/{raw_files})"
        )


def summarize_all_files(worker_count=8):
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
            cp.print_red(f"Erreur lors du traitement du fichier {file_path}: {e}")

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

    calculate_success_rate()


def normalize_user_input(user_input: str):
    user_input = user_input.lower()
    user_input = re.sub(r"[^\w\s]", "", user_input)
    user_input = user_input.translate(str.maketrans("", "", string.punctuation))
    doc = nlp(user_input)
    tokens = [
        token.lemma_
        for token in doc
        if token.text not in stopwords.words("french") and token.is_alpha
    ]
    return " ".join(tokens)


def prepare_text_for_embedding(file_path: str, mode: str):
    if mode == "with_context":
        with_context = True
    else:
        with_context = False

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            if with_context:
                text = f"""éléments de contexte: {' '.join(data['éléments_contexte'])} résumé: {data['résumé']}"""
            else:
                text = f"""résumé: {data['résumé']}"""

            text = text.lower()
            text = re.sub(r"[^\w\s]", "", text)
            text = text.translate(str.maketrans("", "", string.punctuation))
            doc = nlp(text)
            tokens = [
                token.lemma_
                for token in doc
                if token.text not in stopwords.words("french") and token.is_alpha
            ]
            embedded_text = " ".join(tokens)
            return {
                "embedded_text": embedded_text,
                "id": data.get("id"),
                "url": data.get("url"),
                "articles_pertinents": data.get("articles_pertinents"),
                "articles_potentiellement_pertinents": data.get(
                    "articles_potentiellement_pertinents"
                ),
            }

    except (KeyError, Exception, json.JSONDecodeError) as e:
        print(f"Erreur {file_path}: {e}")
        return None


def calculate_and_save_vectors(json_folder: str, pickle_file: str, mode: str):
    results_and_metadata = []
    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

    for json_file in tqdm(
        json_files,
        desc="Analyse des fichiers",
        unit="documents",
        bar_format="[{elapsed} < {remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}",
        colour="blue",
        ascii=" ▖▘▝▗▚▞█",
    ):
        file_path = os.path.join(json_folder, json_file)
        data = prepare_text_for_embedding(file_path, mode)
        if data and data.get("embedded_text"):
            results_and_metadata.append(data)

    results = [item["embedded_text"] for item in results_and_metadata]

    vectorizer = TfidfVectorizer(
        strip_accents="ascii",
        lowercase=True,
        norm="l2",
    )
    tfidf_matrix = vectorizer.fit_transform(results)

    with open(pickle_file, "wb") as f:
        pickle.dump((tfidf_matrix, results_and_metadata, vectorizer), f)


def load_vectors(pickle_file):
    try:
        with open(pickle_file, "rb") as f:
            tfidf_matrix, results_and_metadata, vectorizer = pickle.load(f)
        return tfidf_matrix, results_and_metadata, vectorizer
    except FileNotFoundError:
        return None, None, None


def check_if_vectors_exist(pickle_file: str, mode: str):
    if not os.path.exists(pickle_file):
        calculate_and_save_vectors("./temp_data", pickle_file, mode)


def calculate_similarity(usr_input: str, mode: str):

    if mode == "with_context":
        pickle_file = "./vectors/with_context_elements.pkl"
    else:
        pickle_file = "./vectors/without_context_elements.pkl"

    check_if_vectors_exist(pickle_file, mode)

    tfidf_matrix, results_and_metadata, vectorizer = load_vectors(pickle_file)

    if tfidf_matrix is None:
        print("Vector data not found. Please calculate vectors first.")
        return

    normalized_user_description = normalize_user_input(usr_input)
    user_vector = vectorizer.transform([normalized_user_description])

    cosine_sim_matrix = cosine_similarity(user_vector, tfidf_matrix)

    similarity_list = [
        (i, similarity * 100) for i, similarity in enumerate(cosine_sim_matrix[0])
    ]

    similarity_list.sort(key=lambda x: x[1], reverse=True)
    show_results(similarity_list, results_and_metadata)


def show_results(similarity_list, results_and_metadata):
    high_similarity_count = 0
    low_similarity_count = 0
    ten_similarity_count = 0

    print(
        "-----------------------------------------------------------------------------------"
    )

    for i, similarity in similarity_list:
        metadata = results_and_metadata[i]
        id = metadata.get("id")
        articles = metadata.get("articles_pertinents", [])
        articles_potentiellement_pertinents = metadata.get(
            "articles_potentiellement_pertinents", []
        )
        url = metadata.get("url", "")

        if similarity > 65:
            print(f"% similarité avec doc ID: {id}", end=" -> ")
            cp.print_green(f"{similarity:.2f}%")
            high_similarity_count += 1
            if high_similarity_count > 10000:
                continue

        elif similarity < 65 and similarity > 35 and low_similarity_count < 5:
            print(f"% similarité avec doc ID: {id}", end=" -> ")
            cp.print_yellow(f"{similarity:.2f}%")
            low_similarity_count += 1

        elif abs(similarity - 25) < 1 and ten_similarity_count < 1:
            print(f"% similarité avec doc ID: {id}", end=" -> ")
            cp.print_red(f"{similarity:.2f}%")
            ten_similarity_count += 1

        else:
            continue

        if i < len(results_and_metadata):
            print("Articles pertinents: ")
            for article in articles:
                cp.print_cyan(str(article))
            if articles_potentiellement_pertinents:
                print("Articles potentiellement pertinents: ")
                for article in articles_potentiellement_pertinents:
                    cp.print_cyan(str(article))
            print("URL: ", end=" ")
            cp.print_magenta(url)

        print(
            "-----------------------------------------------------------------------------------"
        )


def find_recommendations():

    usr_input = input(
        "Donnez des éléments de contexte (licenciement, blessure, excés de pouvoir, etc.). Si vous en avez pas, tapez la touche Entrer: "
    )
    if usr_input == "":
        mode = "with_context"
    else:
        mode = "without_context"

    usr_input = input("Donnez un résumé de la situation: ")

    if usr_input == "":
        cp.print_red("Le résumé ne peut pas être vide.")
        return

    calculate_similarity(usr_input, mode)
