import pymupdf
import os
import json
from utils.scrape_utils.scrape_utils import ensure_folder_exists
import utils.terminal_utils.terminal_utils as tu


def extract_text_from_pdf(pdf_path: str, output_path: str):
    doc = pymupdf.open(pdf_path)
    out = open(output_path, "wb")
    for page in doc:
        text = page.get_text()
        out.write(text.encode("utf-8"))
    out.close()
    doc.close()


def print_text_from_pdf(pdf_path: str):
    doc = pymupdf.open(pdf_path)
    for page in doc:
        text = page.get_text()
        print(text)
        print("\f")


def extract_text_from_txt(txt_path: str) -> str:
    try:
        with open(txt_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Errore: il file {txt_path} non esiste.")
        return ""
    except Exception as e:
        print(f"Errore durante la lettura del file {txt_path}: {e}")
        return ""


def save_text_to_txt(text: str, output_path: str) -> bool:
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            print(f"Errore nella creazione della cartella {output_dir}: {e}")
            return False

    try:
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(text)
            print(f"Successfully saved in file: {output_path}")
        return True
    except Exception as e:
        tu.print_red(f"Errore durante il salvataggio del file {output_path}: {e}")
        return False


def save_response_as_json(response):
    if not ensure_folder_exists("donnees_json"):
        tu.print_red("Failed to create output folder.")
        return

    try:
        response_data = json.loads(response)

        if isinstance(response_data, list) and len(response_data) > 0:
            document_id = response_data[0].get("identifiant_document", "unknown_doc")

        json_file = os.path.join("donnees_json", f"{document_id}.json")

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(response_data, f, ensure_ascii=False, indent=4)

        # tu.print_blue(f"Risposta salvata in: {json_file}")
        return True

    except (Exception, json.JSONDecodeError) as e:
        tu.print_red(f"Errore durante il salvataggio del file JSON: {e}")
        return False


def extract_from_json(json_path=None, json_folder="donnees_json"):

    results = []

    json_folder = "./donnees_json"

    if not os.path.exists(json_folder):
        tu.print_red(f"Le dossier {json_folder} n'existe pas.")
        return results

    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

    if not json_files:
        tu.print_yellow(f"Le dossier {json_folder} est vide")
        return results

    for json_file in json_files:
        file_path = os.path.join(json_folder, json_file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append(data)

        except (Exception, json.JSONDecodeError) as e:
            tu.print_red(f"Erreur lecture du JSON du fichier {file_path}: {e}")
            continue

    return results


def extract_metadata(data):
    metadata = {
        "lien_internet": data["lien_internet"],
        "identifiant_document": data["identifiant_document"],
        "date": data["date"],
        "décision": data["décision"],
    }
    return metadata


def extract_combined_text(data):
    combined_text = f"""
    Context: {', '.join(data['éléments_contexte'])}.
    Decision: {data['décision']}.
    Summary: {data['résumé']}.
    Articles: {', '.join(data['articles_pertinents'])}.
    """
    return combined_text
