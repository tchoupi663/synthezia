"""
This module provides utilities for processing legal documents, extracting structured information,
and calculating similarities between user inputs and preprocessed legal documents.
Classes:
    Parties: Represents the plaintiffs and defendants in a legal case.
    Arguments: Represents the arguments presented by the plaintiffs and defendants.
    MontantsFinanciers: Represents financial amounts such as claims and damages.
    DocumentJuridique: Represents a structured legal document with various attributes.
    UserPrompt: Represents a structured user input for legal document processing.
Functions:
    save_to_json(data: dict, id: str):
        Saves a dictionary to a JSON file in the "temp_data" directory.

    read_json_content(file_path: str):
        Reads and parses JSON content from a file.

    summarize_with_ai(data: dict):
        Summarizes a legal document using an AI model and extracts structured information.

    preprocess_text(text: str) -> str:
        Preprocesses text by normalizing, lemmatizing, and removing stopwords.

    prepare_text_for_embedding(file_path: str):
        Prepares text from a JSON file for embedding by weighting and preprocessing its components.

    prepare_user_prompt_for_embedding(data: str):
        Prepares user input for embedding by weighting and preprocessing its components.

    summarize_all_files(worker_count=8):
        Summarizes all files in the "raw_data" directory using multithreading.

    summarize_user_prompt(usr_input: str):
        Summarizes user input into a structured format using an AI model.

    calculate_and_save_vectors(json_folder: str, pickle_file: str):
        Calculates and saves TF-IDF vectors for preprocessed legal documents.

    load_vectors(pickle_file: str):
        Loads precomputed TF-IDF vectors and metadata from a pickle file.

    check_if_vectors_exist(pickle_file: str):
        Checks if vector data exists, and if not, calculates and saves it.

    calculate_similarity(usr_input: str):
        Calculates similarity between user input and preprocessed legal documents.

    show_results(similarity_list, results_and_metadata):
        Displays similarity results and metadata for matching legal documents.

    find_recommendations():
        Prompts the user for a legal summary and finds recommendations based on similarity.
"""

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
import utils.colorama_utils as cp
import string
import pickle
from typing import List, Optional
import copy


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


def read_json_content(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            data = json.load(file)
            return data
    except (Exception, json.JSONDecodeError) as e:
        return None


def summarize_with_ai(data: dict):
    """
    Summarizes a judicial document using an AI model and extracts structured legal information.

    Args:
        data (dict): A dictionary containing the following keys:
            - "texte" (str): The content of the judicial document to be summarized. Defaults to "N/A" if not provided.
            - "id" (str): The unique identifier for the document. Defaults to "N/A" if not provided.
            - "url" (str): The URL associated with the document. Defaults to "N/A" if not provided.

    Returns:
        None: The function processes the AI-generated response, extracts structured data, and saves it as a JSON file.

    Notes:
        - The function uses a pre-configured AI model to generate structured legal information in JSON format.
        - The JSON structure includes fields such as "solution", "parties", "contexte", "faits", "problème_juridique",
          "arguments", "motifs_décision", "articles_cités", "articles_suggérés", and "montants_financiers".
        - If the AI response is valid JSON, the data is enriched with the document ID and URL before being saved.
        - If the AI response is not valid JSON or if no response is received, the function handles these cases gracefully.
    """
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
    """
    Preprocesses the input text by performing the following steps:
    1. Converts the text to lowercase.
    2. Removes all non-alphanumeric characters and punctuation.
    3. Tokenizes the text using a natural language processing (NLP) library.
    4. Lemmatizes the tokens to their base forms.
    5. Filters out stopwords in French and non-alphabetic tokens.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text as a single string of space-separated tokens.
    """
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


def prepare_text_for_embedding(file_path: str):
    """
    Prepares text data from a JSON file for embedding by preprocessing and weighting
    different sections of the text based on their importance.

    Args:
        file_path (str): The path to the JSON file containing the text data.

    Returns:
        dict: A dictionary containing the following keys:
            - "embedded_text" (str): The combined and weighted preprocessed text.
            - "id" (str or None): The ID of the document, if available in the JSON data.
            - "url" (str or None): The URL of the document, if available in the JSON data.
            - "articles_cités" (list or None): A list of cited legal articles, if available.
            - "articles_suggérés" (list or None): A list of suggested legal articles, if available.

        None: If an error occurs during processing (e.g., missing keys, invalid JSON).

    Raises:
        KeyError: If a required key is missing in the JSON data.
        json.JSONDecodeError: If the file content is not valid JSON.
        Exception: For any other unexpected errors.

    Notes:
        - The function preprocesses and weights the following sections:
            - "contexte" (context): Weighted by `context_weight`.
            - "faits" (facts): Weighted by `faits_weight`.
            - "problème_juridique" (legal problem) and "arguments": Combined and weighted by `problème_juridique_weight`.
            - "motifs_décision" (decision justification): Weighted by `motifs_décision_weight`.
            - "articles_cités" (cited articles): Weighted by `articles_cités_weight`.
        - Preprocessing is performed using the `preprocess_text` function (assumed to be defined elsewhere).
    """

    context_weight = 4
    faits_weight = 4
    problème_juridique_weight = 2
    motifs_décision_weight = 4
    articles_cités_weight = 2

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
    """
    Prepares a user prompt for embedding by processing and weighting various
    components of the input data.

    Args:
        data (str): A dictionary containing the following keys:
            - "contexte" (list of str): Contextual information.
            - "faits" (str): Factual summary of the case.
            - "problème_juridique" (str): Legal problem or issue.
            - "arguments" (dict): Arguments presented by the parties, with keys:
                - "demandeur" (list of str or str): Arguments from the claimant.
                - "défendeur" (list of str or str): Arguments from the defendant.
            - "motifs_décision" (str): Justification or reasoning for the decision.
            - "articles_cités" (list of str): Legal articles cited in the case.
            - "id" (optional): Identifier for the data.
            - "url" (optional): URL associated with the data.

    Returns:
        dict: A dictionary containing:
            - "embedded_text" (str): The processed and weighted text ready for embedding.
            - "id": The identifier from the input data (if provided).
            - "url": The URL from the input data (if provided).
            - "articles_cités": The cited legal articles from the input data.
    """

    context_weight = 4
    faits_weight = 4
    problème_juridique_weight = 2
    motifs_décision_weight = 4
    articles_cités_weight = 2

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


def summarize_all_files(worker_count=8):
    """
    Summarizes all JSON files in the "./raw_data" directory using AI.

    This function reads all files in the "./raw_data" directory, processes each JSON file
    to extract specific fields, and then summarizes the content using an AI-based summarization
    function. The processing is done concurrently using a thread pool.

    Args:
        worker_count (int, optional): The number of worker threads to use for concurrent
            file processing. Defaults to 8.

    Raises:
        Exception: If an error occurs while processing a file, it is logged but does not
            interrupt the execution of the function.

    Notes:
        - The JSON files are expected to contain the following fields:
            - "id": A unique identifier for the document.
            - "solution": The decision or solution text.
            - "text": The main content of the document.
            - "url": The source URL of the document.
        - If any of these fields are missing, a default value of "N/A" is used.
        - Progress is displayed using a progress bar with the `tqdm` library.

    Example:
        summarize_all_files(worker_count=4)
    """
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
            colour="blue",
            ascii=" ▖▘▝▗▚▞█",
        ) as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    pbar.write(f"Error processing file {futures[future]}: {e}")
                finally:
                    pbar.update(1)


def summarize_user_prompt(usr_input: str):
    """
    Summarizes a user's legal prompt by extracting structured information in JSON format.
    This function processes the user's input to extract key legal details such as
    decision type, legal context, facts, legal issues, arguments, reasoning, cited articles,
    and financial amounts. It uses a pre-configured model to generate the structured output
    and provides progress updates using a progress bar.
    Args:
        usr_input (str): The user's input text containing legal information to be summarized.
    Returns:
        dict or None: A dictionary containing the extracted legal information in the specified
        JSON format, or None if the response could not be processed.
    JSON Output Format:
            "contexte": ["Concepts juridiques clés (max 5 termes)", ...],
    Progress Bar Stages:
        1. Vérification de l'entrée utilisateur
        2. Envoi de la requête
        3. Traitement de la réponse
        4. Dernière vérification
    Notes:
        - The function uses a pre-configured model (`gemini-2.0-flash`) to generate the content.
        - If the response cannot be parsed as JSON, the function returns None.
        - The function ensures that the extracted information adheres to specific rules for
          formatting and content extraction.
    """

    with tqdm(
        total=5,
        desc="Traitement",
        colour="blue",
        bar_format="[{elapsed}] {n_fmt}/{total_fmt} | {l_bar}{bar}",
    ) as pbar:

        pbar.set_description("Vérification de l'entrée utilisateur")
        pbar.update(1)

        generate_content_config = types.GenerateContentConfig(
            temperature=1.8,
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048,
            response_mime_type="application/json",
            response_schema=UserPrompt,
            system_instruction="""
        Extrayez les informations juridiques du prompt utilisateur en suivant CE FORMAT JSON. Si le texte ne contient pas assez d'informations, ajoutez des elements qui pourraient booster la recherche de similarité avec les vecteurs existants, mais ne modifiez pas le sens du texte. 

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
   - Si il n'y a pas de faits, ajouter non spécifié

3. **Montants** :  
   - Convertir les écritures littérales → chiffres ("cinquante mille" → 50000)  
   - Formater sans espaces/€ : "50 000" → 50000  

4. **Contextes juridiques** :  
   - Privilégier les notions codifiées : "manquement à l'obligation de conseil" plutôt que "mauvaise advice"  
   
5. "articles_cités": 
    - Inclure les articles explicitement mentionnés dans le texte, dans le format "article 1234 du code civil" ou "article L. 123-4 du code de la consommation".

        ---
        
Exemple de texte d'entrée :
Les échanges de mails et le rôle central dans le système de contrats fictifs, caractérisent-ils suffisamment les délits de détournement de fonds publics et d'abus de confiance et de complicité de ce délit?
Exemple de texte de sortie DANS CE CAS:
{'solution': 'Non spécifié', 'contexte': ['Détournement de fonds publics', 'Abus de confiance', 'Complicité de délit', 'Contrats fictifs'], 'faits': 'Echanges de mails et rôle central dans un système de contrats fictifs', 'problème_juridique': "Les échanges de mails et le rôle central dans un système de contrats fictifs caractérisent-ils suffisamment les délits de détournement de fonds publics, d'abus de confiance et de complicité de ces délits ?", 'arguments': {'demandeur': 'Non spécifié', 'défendeur': 'Non spécifié'}, 'motifs_décision': 'Non spécifié', 'articles_cités': [], 'montants_financiers': {'réclamation': None, 'dommages_intérêts': None}}

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

        pbar.set_description("Dernière vérification")
        pbar.update(1)
        return processed_data


def calculate_and_save_vectors(json_folder: str, pickle_file: str):
    """
    Processes JSON files in a specified folder, calculates TF-IDF vectors for the embedded text,
    and saves the resulting matrix, metadata, and vectorizer to a pickle file.

    Args:
        json_folder (str): Path to the folder containing JSON files to process.
        pickle_file (str): Path to the output pickle file where the results will be saved.

    The function performs the following steps:
    1. Reads all JSON files in the specified folder.
    2. Extracts and processes text data for embedding using `prepare_text_for_embedding`.
    3. Filters out files without valid embedded text.
    4. Computes a TF-IDF matrix for the embedded text using `TfidfVectorizer`.
    5. Saves the TF-IDF matrix, metadata, and vectorizer to the specified pickle file.

    Notes:
        - The TF-IDF vectorizer is configured to strip accents, convert text to lowercase,
          and use raw term frequencies without normalization.
    """
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


def calculate_similarity(usr_input: str):

    pickle_file = "./vectors/vector.pkl"

    check_if_vectors_exist(pickle_file)

    tfidf_matrix_original, results_and_metadata, vectorizer_original = load_vectors(
        pickle_file
    )

    if tfidf_matrix_original is None:
        cp.print_red("Vector data not found. Please calculate vectors first.")
        return

    summarized_usr_input = summarize_user_prompt(usr_input)

    normalized_user_description = prepare_user_prompt_for_embedding(
        summarized_usr_input
    )
    usr_embedded_text = normalized_user_description.get("embedded_text")

    vectorizer = copy.deepcopy(vectorizer_original)
    tfidf_matrix = copy.deepcopy(tfidf_matrix_original)

    user_vector = vectorizer.transform([usr_embedded_text])

    cosine_sim_matrix = cosine_similarity(user_vector, tfidf_matrix)

    similarity_list = [
        (i, similarity * 100) for i, similarity in enumerate(cosine_sim_matrix[0])
    ]

    similarity_list.sort(key=lambda x: x[1], reverse=True)

    show_results(similarity_list, results_and_metadata)


def show_results(similarity_list, results_and_metadata):
    """
    Displays the results of a similarity analysis along with associated metadata.

    This function iterates through a list of similarity scores and their corresponding metadata,
    filtering and categorizing the results based on similarity thresholds. It prints the results
    in a formatted manner, highlighting high, medium, and specific similarity scores, and includes
    relevant metadata such as cited and suggested articles, as well as URLs.

    Args:
        similarity_list (list of float): A list of similarity scores.
        results_and_metadata (list of dict): A list of dictionaries containing metadata for each result.
            Each dictionary may include the following keys:
                - "id" (str): The document ID.
                - "articles_cités" (list of str): A list of cited articles.
                - "articles_suggérés" (list of str): A list of suggested articles.
                - "url" (str): A URL associated with the result.

    Behavior:
        - Filters out results that reference excluded articles (e.g., "article 567-1-1" or "article 1014").
        - Categorizes results into:
            - High similarity (> 40%).
            - Medium similarity (12% < similarity <= 40%).
            - Specific similarity (~10%).
        - Limits the number of displayed results for medium and specific similarity categories.
        - Prints metadata such as cited and suggested articles, and URLs for each result.

    Notes:
        - Skips processing if the similarity score does not meet any of the defined thresholds.
    """
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

        found_excluded_article = False
        if articles_cités:
            for article in articles_cités:
                if article.startswith("article 567-1-1"):
                    found_excluded_article = True
                    break
                elif article.startswith("article 1014"):
                    found_excluded_article = True
                    break

        if found_excluded_article:
            continue

        articles_suggérés = metadata.get("articles_suggérés", [])
        url = metadata.get("url", "")

        if similarity > 40:
            print(f"% similarité avec doc ID: {id}", end=" -> ")
            cp.print_green(f"{similarity:.2f}%")
            high_similarity_count += 1
            if high_similarity_count > 10000:
                continue

        elif similarity < 40 and similarity > 12 and low_similarity_count < 15:

            print(f"% similarité avec doc ID: {id}", end=" -> ")
            cp.print_yellow(f"{similarity:.2f}%")
            low_similarity_count += 1

        elif abs(similarity - 10) < 1 and ten_similarity_count < 5:
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
    cp.print_cyan("             (1) le problème juridique précis,")
    cp.print_cyan("             (2) les articles de loi applicables,")
    cp.print_cyan("             (3) les faits-clés (montants, durées, sanctions).")
    cp.print_cyan(
        "             (4) Utilisez des termes techniques (détention provisoire, infractions aux législations, etc.)\n"
    )

    usr_input = input("          -> ")

    if usr_input == "":
        cp.print_red("Le résumé ne peut pas être vide.")
        return

    calculate_similarity(usr_input)
