import requests
import utils.terminal_utils.terminal_utils as cp
from dotenv import load_dotenv
import os
import json
from tqdm import tqdm


def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Cartella creata: {folder_path}")


def save_to_json(data, file_name):
    file_path = os.path.join("raw_data", file_name)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        cp.print_red(f"Errore durante il salvataggio del file JSON: {e}")


def clean_up_text(text):
    text = text.replace("\n", "")
    target_word = "AU NOM DU PEUPLE FRANÇAIS _________________________   "
    char_count = text.find(target_word) + len(target_word)
    text = text[char_count:]
    return text


def retrieve(i: int):
    params = {
        "batch_size": "10",
        "batch": {i},
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
    total_batches = 1000
    for i in tqdm(
        range(total_batches),
        desc="Récuperation des données",
        unit="documents",
        bar_format="[{elapsed} < {remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}",
        colour="green",
        ascii=" ▖▘▝▗▚▞█",
    ):
        retrieve(i)
