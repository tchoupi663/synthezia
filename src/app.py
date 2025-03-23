#!/usr/bin/env python3

import utils.terminal_utils.terminal_utils as cp
import utils.scrape_utils.scrape_utils as sc
import utils.ai_utils.ai_utils as ai
import utils.terminal_utils.terminal_utils as cp
from sys import exit


def display_message():
    cp.print_red("You selected Option 1: Display a message.")
    print("This is a temporary message.")


def scrape():
    num_pages = int(
        input("Combien de documents de jurisprudence voulez-vous extraire? ")
    )
    sc.extract_from_website(num_pages)


def analyse():
    ai.traitement_textes()


def recommandations():
    cp.print_blue("Donner des recommandations")


def exit_program():
    cp.print_blue("Quitter...")
    exit()


def main():
    cp.print_magenta(
        """
  _________                __   .__                     .__         
 /   _____/___.__.  ____ _/  |_ |  |__    ____  ________|__|_____   
 \_____  \<   |  | /    \\   __\|  |  \ _/ __ \ \___   /|  |\__  \  
 /        \\___  ||   |  \|  |  |   Y  \\  ___/  /    / |  | / __ \_
/_______  // ____||___|  /|__|  |___|  / \___  >/_____ \|__|(____  /
        \/ \/          \/            \/      \/       \/         \/
"""
    )
    while True:
        cp.print_magenta("\n--- Menu ---")
        print("1. Display a message")
        print("2. Extraction de documents de jurisprudence")
        print("3. Extraction des données importantes")
        print("4. Donner des recommendantions")
        print("9. Quitter")

        choice = input("Selectionnez une options (1-9): ")

        if choice == "1":
            display_message()
        elif choice == "2":
            scrape()
        elif choice == "3":
            analyse()
        elif choice == "4":
            display_message()
        elif choice == "9":
            exit_program()
        else:
            cp.print_red("Invalide")


if __name__ == "__main__":
    main()
