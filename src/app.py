#!/usr/bin/env python3

import utils.terminal_utils.terminal_utils as cp
import utils.file_utils.file_utils as pu
import utils.scrape_utils.scrape_utils as sc
import utils.ai_utils.ai_utils as ai


def display_message():
    cp.print_red("You selected Option 1: Display a message.")
    print("This is a temporary message.")


def scrape():
    num_pages = int(
        input("Combien de documents de jurisprudence voulez-vous extraire? ")
    )
    sc.extract_from_website(num_pages)


def analyse():
    ai.traitement_dossier_textes()


def exit_program():
    cp.print_blue("Quitter...")
    exit()


def main():
    while True:
        cp.print_magenta("\n--- Menu ---")
        print("1. Display a message")
        print("2. Extraction de documents de jurisprudence")
        print("3. Extraction des données importantes")
        print("9. Quitter")

        choice = input("Selectionnez une options (1-9): ")

        if choice == "1":
            display_message()
        elif choice == "2":
            scrape()
        elif choice == "3":
            analyse()
        elif choice == "9":
            exit_program()
        else:
            cp.print_red("Invalide")


if __name__ == "__main__":
    main()
