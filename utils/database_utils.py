"""
This module provides utility functions for interacting with a legal database API,
processing retrieved data, and saving it in JSON format.

Functions:
    ensure_folder_exists(folder_path):
        Ensures that a specified folder exists, creating it if necessary.

    save_to_json(data, file_name):
        Saves the given data to a JSON file in the "temp_raw" directory.

    clean_up_text(text):
        Cleans up the input text by removing unwanted characters and trimming
        it to exclude a specific target phrase.

    retrieve(i: int):
        Retrieves a batch of legal case data from the API based on the given
        batch index, processes the data, and saves it as JSON files.

    retrieve_all():
        Iterates through all batches of data, retrieves them using the `retrieve`
        function, and displays a progress bar during the process.
"""

import requests
import utils.colorama_utils as cp
from dotenv import load_dotenv
import os
import json
from tqdm import tqdm


def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Cartella creata: {folder_path}")


def save_to_json(data, file_name):
    file_path = os.path.join("temp_raw", file_name)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        cp.print_red(f"Errore durante il salvataggio del file JSON: {e}")


def clean_up_text(text):
    """
    Cleans up the input text by removing newline characters and truncating it
    from a specific target phrase onward.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text with newline characters removed and truncated
        from the end of the target phrase.
    """
    text = text.replace("\n", "")
    target_word = "AU NOM DU PEUPLE FRANÇAIS _________________________   "
    char_count = text.find(target_word) + len(target_word)
    text = text[char_count:]
    return text


def retrieve(i: int):
    """
    Retrieves case data from the Judilibre API based on the specified batch index.

    Args:
        i (int): The batch index to retrieve data for.

    Raises:
        requests.exceptions.RequestException: If the API request fails.
        KeyError: If the expected keys are missing in the API response.

    Environment Variables:
        JUDILIBRE_KEY: The API key required for authentication with the Judilibre API.

    Notes:
        - The function constructs a set of parameters to query the Judilibre API.
        - The API response is expected to contain case data in JSON format.
        - Each case's data is processed, cleaned, and saved as a JSON file.
        - If no results are found, a message is printed to indicate this.

    Example:
        retrieve(1)
    """
    params = {
        "batch_size": "10",
        "batch": {i},
        "type": "arret",
        "chamber": "cr",  # Chambre criminelle
        "jurisdiction": "cc",
        "publication": "b",
    }

    load_dotenv()
    api_key = os.getenv("JUDILIBRE_KEY")
    headers = {
        "KeyId": api_key,
    }

    response = requests.get(
        "https://sandbox-api.piste.gouv.fr/cassation/judilibre/v1.0/export",
        params=params,
        headers=headers,
    )

    data = response.json()
    if data:
        for case in data["results"]:
            case_data = {
                "number": case.get("number", "N/A"),
                "id": case.get("id", "N/A"),
                "solution": case.get("solution", "N/A"),
                "jurisdiction": ("Cour de cassation"),
                "text": clean_up_text(case.get("text", "N/A")),
                "url": f"https://www.courdecassation.fr/decision/{case.get('id', 'N/A')}",
            }
            save_to_json(case_data, f"CAS_{case.get('id', 'N/A')}.json")

    else:
        cp.print_red("Nessun risultato trovato nella risposta.")


def retrieve_all():
    """
    Retrieves data in batches and displays a progress bar.

    This function iterates through a predefined number of batches (`total_batches`)
    and calls the `retrieve` function for each batch. A progress bar is displayed
    using the `tqdm` library to indicate the progress of the data retrieval process.

    Progress Bar Details:
    - Description: "Récuperation des données" (French for "Data Retrieval").
    - Unit: "documents".
    - Bar Format: Displays elapsed time, remaining time, current progress, and rate.
    - Colour: Blue.
    - ASCII: Custom characters for the progress bar.

    """
    total_batches = 1000
    for i in tqdm(
        range(total_batches),
        desc="Récuperation des données",
        unit="documents",
        bar_format="[{elapsed} < {remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}",
        colour="blue",
        ascii=" ▖▘▝▗▚▞█",
    ):
        retrieve(i)
