import requests
from bs4 import BeautifulSoup
import utils.colour_utils.colour_printer as cp
import os


base_url = "https://www.legifrance.gouv.fr/juri/id/"
changing_part = "JURITEXT000051311000"
query_params = "?page=1&pageSize=100&searchField=ALL&searchType=ALL&sortValue=DATE_DESC&tab_selection=juri&typePagination=DEFAULT"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Safari/537.36"
}
output_folder = "donnees"


def generate_document_id(current_number):
    return f"JURITEXT{current_number:012}"


def fetch_page(page_number):
    current_id = generate_document_id(51311794 + page_number)
    global file_id
    url = base_url + current_id + query_params
    print(f"Fetching URL: {url}")
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        file_id = 51311794 + page_number
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
    cp.print_blue(f"Starting extraction of {num_pages} documents from the website...")
    page = 0
    documents_read = 0
    while documents_read < num_pages:
        soup = fetch_page(page)
        page += 1
        if soup:
            content = extract_content(soup)
            if content:
                copy_to_file(content)
                documents_read += 1
            else:
                cp.print_yellow("No content found on the page.")
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
        cp.print_red("")
        return

    filename = f"JURITEXT{file_id:012}.txt"
    file_path = os.path.join(output_folder, filename)

    with open(file_path, "w", encoding="utf-8") as file:
        for text in content:
            file.write(text)

    cp.print_green(
        f"Contenuto salvato nel file: {filename} nella cartella: {output_folder}"
    )
