import requests
from bs4 import BeautifulSoup
import utils.terminal_utils.terminal_utils as cp
import os
import sys

base_url = "https://www.legifrance.gouv.fr/juri/id/"
changing_part = 51151385  # 51311677 #51284058 #51243729 #51151385 #000047781047
query_params = "?page=1&pageSize=100&searchField=ALL&searchType=ALL&sortValue=DATE_DESC&tab_selection=juri&typePagination=DEFAULT"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Safari/537.36"
}
output_folder = "donnees"


def generate_document_id(current_number):
    return f"JURITEXT{current_number:012}"


def fetch_page(page_number):
    current_id = generate_document_id(changing_part + page_number)
    global file_id
    url = base_url + current_id + query_params
    global address
    address = url
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        file_id = changing_part + page_number
        return BeautifulSoup(response.content, "html.parser")
    else:
        print(f"Failed to fetch page: {response.status_code}")
        return None


def extract_content(soup):
    if soup is None:
        return []

    content = []
    main_content = soup.find("div", class_="content-page")
    if main_content:
        content = main_content.get_text(separator="\n", strip=True)

    return content


def extract_from_website(num_pages):
    cp.colorama.init(autoreset=True)
    print(
        cp.Fore.BLUE
        + f"Starting extraction of {num_pages} documents from the website..."
    )
    page = 0
    success = 0
    documents_read = 0
    empty_pages = 0
    while documents_read < num_pages:
        soup = fetch_page(page)
        page += 1
        if empty_pages > 20:
            print(
                f"\n{cp.Fore.RED}Trop de pages vides. Arrêt de l'extraction.{cp.Style.RESET_ALL}"
            )
            success = 0
            break
        if soup:
            content = extract_content(soup)
            if content:
                copy_to_file(content)
                documents_read += 1
                sys.stdout.write("\r" + " " * 80 + "\r")
                sys.stdout.flush()
                print(
                    f"\rDocuments extraits: {cp.Fore.BLUE}{documents_read}/{num_pages}{cp.Style.RESET_ALL}",
                    end="\r",
                )
                success = 1
            else:
                empty_pages += 1
                sys.stdout.write("\r" + " " * 80 + "\r")
                sys.stdout.flush()
                print(
                    f"\rPage web vide {cp.Fore.YELLOW}{documents_read}/{num_pages}{cp.Style.RESET_ALL}",
                    end="\r",
                )

    if success == 1:
        print(f"{cp.Fore.GREEN}Extraction des données réussie.{cp.Style.RESET_ALL}")
    else:
        print(
            f"{cp.Fore.RED}Extraction des données échouée. Documents extraits: {documents_read}/{num_pages} {cp.Style.RESET_ALL}"
        )
    return True


def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            cp.print_magenta(f"Cartella creata: {folder_path}")
        except Exception as e:
            cp.print_red(f"Errore nella creazione della cartella: {e}")
            return False
    return True


def copy_to_file(content):
    if not ensure_folder_exists(output_folder):
        cp.print_red("Failed to create output folder.")
        return

    filename = f"JURITEXT{file_id:012}.txt"
    file_path = os.path.join(output_folder, filename)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(address + "\n")
        for text in content:
            file.write(text)

    cp.print_green(
        f"\rContenuto salvato nel file: {filename} nella cartella: {output_folder}"
    )
