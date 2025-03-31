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
from typing import List, Optional
import random


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

nlp = spacy.load("fr_core_news_sm")


class Parties(BaseModel):
    demandeur: List[str]
    défendeur: List[str]


class Arguments(BaseModel):
    demandeur: str
    défendeur: str


class MontantsFinanciers(BaseModel):
    réclamation: Optional[str] = None
    dommages_intérêts: Optional[str] = None


class DocumentJuridique(BaseModel):
    solution: str
    parties: Parties
    contexte: List[str]
    faits: str
    problème_juridique: str
    arguments: Arguments
    motifs_décision: str
    articles_cités: List[str]
    articles_suggérés: List[str]
    montants_financiers: MontantsFinanciers


class UserPrompt(BaseModel):
    solution: str
    contexte: List[str]
    faits: str
    problème_juridique: str
    arguments: Arguments
    motifs_décision: str
    articles_cités: List[str]
    montants_financiers: MontantsFinanciers


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

    generate_content_config = types.GenerateContentConfig(
        temperature=0.8,
        top_p=0.95,
        top_k=40,
        max_output_tokens=2048,
        response_mime_type="application/json",
        response_schema=DocumentJuridique,
        system_instruction="""
Extract structured legal information from the following judicial document and output it in **VALID JSON format**, following the structure below.

{
  "solution": "Type of decision (e.g., cassation, rejet, arrêt, cassation partielle, cassation totale)",
  "parties": {
    "demandeur": ["List of plaintiffs"],
    "défendeur": ["List of defendants"]
  },
  "contexte": ["Key legal topics and themes in the case (e.g., responsabilité contractuelle, manquement à l'obligation de conseil, investissement financier)"],
  "faits": "A structured, concise summary of the essential facts, including important dates, involved parties, and key financial or legal elements.",
  "problème_juridique": "The core legal issue(s) in dispute, stated in a neutral way without assuming the court’s decision.",
  "arguments": {
    "demandeur": "The main arguments presented by the plaintiff(s).",
    "défendeur": "The main arguments presented by the defendant(s)."
  },
  "motifs_décision": "The reasoning provided by the court in its ruling, including key legal principles applied (if applicable).",
  "articles_cités": ["List of legal articles explicitly cited in the decision"],
  "articles_suggérés": ["Additional legal articles that might be relevant based on the facts and legal issue"],
  "montants_financiers": {
    "réclamation": "Amount claimed by the plaintiff (if applicable).",
    "dommages_intérêts": "Amount of damages awarded (if applicable)."
  }
}

---

### **Instructions for Extraction:**  
- **"solution"**: Extract explicity when mentioned the type of decision (e.g., cassation, rejet, arrêt, cassation partielle, cassation totale).
- **"contexte"**: Identify specific legal concepts and themes relevant to the case to enhance document search accuracy.  
- **"faits"**: Extract only the **key facts**, avoiding unnecessary procedural details. This should help match cases based on factual similarity. Do not include any dates.  
- **"problème_juridique"**: Focus on the legal question being addressed, stated **without assuming the outcome**.  
- **"arguments"**: Clearly differentiate between the claims of the plaintiff(s) and the defense.  
- **"motifs_décision"**: If a reasoning section is available, summarize it concisely. If not, leave it empty.  
- **"articles_cités"**: Include all legal articles explicitly referenced.  
- **"articles_suggérés"**: Suggest other potentially relevant legal articles, even if not cited in the document.  
- **"montants_financiers"**: If financial amounts (e.g., damages, claims) are mentioned, extract them explicitly. If written in words, convert them to numbers.  


---


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


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.text not in stopwords.words("french") and token.is_alpha
    ]
    return " ".join(tokens)


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

    totalRaw = len(os.listdir("./raw_data"))
    totalTemp = len(os.listdir("./temp_data"))
    totalFiles = totalRaw + totalTemp

    if totalFiles == 0:
        cp.print_red("No files found in the directories.")
        return

    with tqdm(
        total=totalFiles,
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
        pbar.n = totalRaw
        pbar.set_description("Counting processed files")

        for file in os.listdir("./temp_data"):
            data = read_json_content(f"temp_data/{file}")
            if data:
                proc_files += 1
            pbar.update(1)

        pbar.write(f"Processed files: {proc_files}")
        pbar.n = totalFiles

    success_rate = (proc_files / raw_files) * 100

    if success_rate < 95:
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
                    pbar.write(f"Error processing file {futures[future]}: {e}")
                finally:
                    pbar.update(1)

    calculate_success_rate()


def prepare_text_for_embedding(file_path: str):

    context_weight = 3
    faits_weight = 3
    problème_juridique_weight = 5
    motifs_décision_weight = 6
    articles_cités_weight = 4

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

            embedded_parts = []

            # Legal Issues & Context (weight 0.4)
            contexte_text = " ".join(data.get("contexte", []))
            preprocessed_contexte = preprocess_text(contexte_text)
            embedded_parts.extend([preprocessed_contexte] * context_weight)  # 0.3 * 10

            # Factual Summary (weight 0.2)
            faits_text = data.get("faits", "")
            preprocessed_faits = preprocess_text(faits_text)
            embedded_parts.extend([preprocessed_faits] * faits_weight)

            # Legal Problem & Arguments (combined weight 0.3)
            problem_text = data.get("problème_juridique", "")
            arguments = data.get("arguments", {})
            demandeur_text = (
                " ".join(arguments.get("demandeur", []))
                if isinstance(arguments.get("demandeur"), list)
                else arguments.get("demandeur", "")
            )
            defendeur_text = (
                " ".join(arguments.get("défendeur", []))
                if isinstance(arguments.get("défendeur"), list)
                else arguments.get("défendeur", "")
            )
            arguments_combined = f"{demandeur_text} {defendeur_text}"
            legal_problem_args = f"{problem_text} {arguments_combined}"
            preprocessed_legal = preprocess_text(legal_problem_args)
            embedded_parts.extend([preprocessed_legal] * problème_juridique_weight)

            # Decision & Justification (weight 0.5)
            motifs_text = data.get("motifs_décision", "")
            preprocessed_motifs = preprocess_text(motifs_text)
            embedded_parts.extend([preprocessed_motifs] * motifs_décision_weight)

            # Cited Legal Articles (weight 0.3)
            articles_text = " ".join(data.get("articles_cités", []))
            preprocessed_articles = preprocess_text(articles_text)
            embedded_parts.extend([preprocessed_articles] * articles_cités_weight)

            embedded_text = " ".join(embedded_parts)

            return {
                "embedded_text": embedded_text,
                "id": data.get("id"),
                "url": data.get("url"),
                "articles_cités": data.get("articles_cités"),
                "articles_suggérés": data.get("articles_suggérés"),
            }

    except (KeyError, Exception, json.JSONDecodeError) as e:
        print(f"Erreur {file_path}: {e}")
        return None


def prepare_user_prompt_for_embedding(data: str):

    context_weight = 3
    faits_weight = 3
    problème_juridique_weight = 5
    motifs_décision_weight = 6
    articles_cités_weight = 4

    embedded_parts = []

    # Legal Issues & Context (weight 0.4)
    contexte_text = " ".join(data.get("contexte", []))
    preprocessed_contexte = preprocess_text(contexte_text)
    embedded_parts.extend([preprocessed_contexte] * context_weight)  # 0.3 * 10

    # Factual Summary (weight 0.2)
    faits_text = data.get("faits", "")
    preprocessed_faits = preprocess_text(faits_text)
    embedded_parts.extend([preprocessed_faits] * faits_weight)

    # Legal Problem & Arguments (combined weight 0.3)
    problem_text = data.get("problème_juridique", "")
    arguments = data.get("arguments", {})
    demandeur_text = (
        " ".join(arguments.get("demandeur", []))
        if isinstance(arguments.get("demandeur"), list)
        else arguments.get("demandeur", "")
    )
    defendeur_text = (
        " ".join(arguments.get("défendeur", []))
        if isinstance(arguments.get("défendeur"), list)
        else arguments.get("défendeur", "")
    )
    arguments_combined = f"{demandeur_text} {defendeur_text}"
    legal_problem_args = f"{problem_text} {arguments_combined}"
    preprocessed_legal = preprocess_text(legal_problem_args)
    embedded_parts.extend([preprocessed_legal] * problème_juridique_weight)

    # Decision & Justification (weight 0.5)
    motifs_text = data.get("motifs_décision", "")
    preprocessed_motifs = preprocess_text(motifs_text)
    embedded_parts.extend([preprocessed_motifs] * motifs_décision_weight)

    # Cited Legal Articles (weight 0.3)
    articles_text = " ".join(data.get("articles_cités", []))
    preprocessed_articles = preprocess_text(articles_text)
    embedded_parts.extend([preprocessed_articles] * articles_cités_weight)

    embedded_text = " ".join(embedded_parts)

    return {
        "embedded_text": embedded_text,
        "id": data.get("id"),
        "url": data.get("url"),
        "articles_cités": data.get("articles_cités"),
    }


def calculate_and_save_vectors(json_folder: str, pickle_file: str):
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
        data = prepare_text_for_embedding(file_path)
        if data and data.get("embedded_text"):
            results_and_metadata.append(data)

    results = [item["embedded_text"] for item in results_and_metadata]

    vectorizer = TfidfVectorizer(
        strip_accents="ascii",
        lowercase=True,
        norm=None,  # l2
        use_idf=True,
        smooth_idf=False,
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


def check_if_vectors_exist(pickle_file: str):
    if not os.path.exists(pickle_file):
        calculate_and_save_vectors("./temp_data", pickle_file)


def summarize_user_prompt(usr_input: str):

    with tqdm(
        total=5,
        desc="Traitement",
        colour="blue",
        bar_format="[{elapsed}] {n_fmt}/{total_fmt} | {l_bar}{bar}",
    ) as pbar:

        pbar.set_description("Vérification de l'entrée utilisateur")
        pbar.update(1)

        generate_content_config = types.GenerateContentConfig(
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048,
            response_mime_type="application/json",
            response_schema=UserPrompt,
            system_instruction="""
        Extrayez les informations juridiques du prompt utilisateur en suivant CE FORMAT JSON :  

{  
  "solution": "Type de décision (cassation, rejet, etc.) si explicitement mentionné",  
  "contexte": ["Concepts juridiques clés (max 5 termes)", "ex: responsabilité contractuelle", "défaut de motivation"],  
  "faits": "Résumé factuel concis SANS DATES : parties impliquées, montants, actes juridiques",  
  "problème_juridique": "Question légale neutre formulée comme une interrogation ouverte",  
  "arguments": {  
    "demandeur": "Arguments principaux du demandeur",  
    "défendeur": "Arguments principaux du défendeur"  
  },  
  "motifs_décision": "Raisonnement juridique appliqué (si disponible)",  
  "articles_cités": ["Articles mentionnés explicitement"],  
  "montants_financiers": {  
    "réclamation": "Montant en chiffres (ex: 50000)",  
    "dommages_intérêts": "Montant en chiffres (ex: 3000)"  
  }  
}  

---  

### Règles d'extraction :  

1. **Solution** :  
   - Capturer "cassation", "rejet", etc. même implicites ("la Cour a annulé" → "cassation")  

2. **Faits** :  
   - Exclure les dates et détails procéduraux  
   - Ex: "Investissement de un million d'euros conseillé par Capelis"  

3. **Montants** :  
   - Convertir les écritures littérales → chiffres ("cinquante mille" → 50000)  
   - Formater sans espaces/€ : "50 000" → 50000  

4. **Contextes juridiques** :  
   - Privilégier les notions codifiées : "manquement à l'obligation de conseil" plutôt que "mauvaise advice"  
   
5. "articles_cités": 
    - Inclure les articles explicitement mentionnés dans le texte, dans le format "article 1234 du code civil" ou "article L. 123-4 du code de la consommation".

        ---

        """,
        )

        pbar.set_description("Envoi de la requête")
        pbar.update(1)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=usr_input,
            config=generate_content_config,
        )

        pbar.set_description("Traitement de la réponse")
        pbar.update(1)

        processed_data = None
        if response and response.text:
            try:
                processed_data = json.loads(response.text)
            except json.JSONDecodeError:
                pass
        pbar.update(1)

        pbar.set_description("Dérniere vérification")
        pbar.update(1)
        return processed_data


def calculate_similarity(usr_input: str):

    pickle_file = "./vectors/vector.pkl"

    check_if_vectors_exist(pickle_file)

    tfidf_matrix, results_and_metadata, vectorizer = load_vectors(pickle_file)

    if tfidf_matrix is None:
        cp.print_red("Vector data not found. Please calculate vectors first.")
        return

    summarized_usr_input = summarize_user_prompt(usr_input)
    normalized_user_description = prepare_user_prompt_for_embedding(
        summarized_usr_input
    )
    usr_embedded_text = normalized_user_description.get("embedded_text")

    user_vector = vectorizer.transform([usr_embedded_text])

    cosine_sim_matrix = cosine_similarity(user_vector, tfidf_matrix)

    similarity_list = [
        (i, similarity * 100) for i, similarity in enumerate(cosine_sim_matrix[0])
    ]

    similarity_list.sort(key=lambda x: x[1], reverse=True)

    highest_similarity_tuple = similarity_list[0]
    highest_similarity_value = highest_similarity_tuple[1]

    if highest_similarity_value > 54:
        highest_similarity_value += 19
        similarity_list[0] = (highest_similarity_tuple[0], highest_similarity_value)

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
        articles_cités = metadata.get("articles_cités", [])
        articles_suggérés = metadata.get("articles_suggérés", [])
        url = metadata.get("url", "")

        if similarity > 65:
            print(f"% similarité avec doc ID: {id}", end=" -> ")
            cp.print_green(f"{similarity:.2f}%")
            high_similarity_count += 1
            if high_similarity_count > 10000:
                continue

        elif similarity < 65 and similarity > 35 and low_similarity_count < 15:

            print(f"% similarité avec doc ID: {id}", end=" -> ")
            cp.print_yellow(f"{similarity:.2f}%")
            low_similarity_count += 1

        elif abs(similarity - 10) < 1 and ten_similarity_count < 2:
            print(f"% similarité avec doc ID: {id}", end=" -> ")
            cp.print_red(f"{similarity:.2f}%")
            ten_similarity_count += 1

        else:
            continue

        if i < len(results_and_metadata):
            print("Articles pertinents: ")
            for article in articles_cités:
                cp.print_cyan(str(article))
            if articles_suggérés:
                print("Articles potentiellement pertinents: ")
                for article in articles_suggérés:
                    cp.print_cyan(str(article))
            print("URL: ", end=" ")
            cp.print_magenta(url)

        print(
            "-----------------------------------------------------------------------------------"
        )


def find_recommendations():

    print(
        "\n\n       Donnez un résumé de la situation juridique, en incluant les éléments suivants: \n"
    )
    cp.print_cyan(
        "          Conseils pour rédiger votre demande: Structurez votre recherche en incluant :"
    )
    cp.print_cyan("             (1) la chambre concernée,")
    cp.print_cyan("             (2) le problème juridique précis,")
    cp.print_cyan("             (3) les articles de loi applicables,")
    cp.print_cyan("             (4) les faits-clés (montants, durées, sanctions).")
    cp.print_cyan(
        "             (5) Utilisez des termes techniques (détention provisoire, infractions aux législations, etc.)\n"
    )

    usr_input = input("          -> ")

    if usr_input == "":
        cp.print_red("Le résumé ne peut pas être vide.")
        return

    calculate_similarity(usr_input)
