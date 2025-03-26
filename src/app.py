#!/usr/bin/env python3

import utils.terminal_utils.terminal_utils as cp
import utils.dbUpdate.dbUpdate as db
import utils.ai_utils.ai_utils as ai
from sys import exit


def display_message():
    cp.print_red("You selected Option 1: Display a message.")
    print("This is a temporary message.")


def retrieve():
    db.retrieve_all()


def analyse():
    ai.traitement_textes()


def recommandations():
    # usr_input = input("Entrez votre description: ")
    # print("user input: ", usr_input)
    usr_input = "L'individu a été victime d'un accident de travail et a subi une blessure à la jambe. Il a été hospitalisé pendant 3 jours et a reçu un traitement médical. L'employeur a refusé de payer les frais médicaux. L'individu a intenté une action en justice contre l'employeur pour obtenir une indemnisation pour les frais médicaux et les dommages et intérêts."
    ai.find_recommandations(usr_input)


def exit_program():
    cp.print_blue("Quitter...")
    exit()


def main():
    cp.print_magenta(
        """
  _________                __   .__                     .__
 /   _____/___.__.  ____ _/  |_ |  |__    ____  ________|__|_____
 \\_____  \\<   |  | /    \\\\   __\\|  |  \\ _/ __ \\ \\___   /|  |\\__  \\
 /        \\\\___  ||   |  \\\\  |  |   Y  \\\\  ___/  /    / |  | / __ \\_
/_______  // ____||___|  /|__|  |___|  / \\___  >/_____ \\|__|(____  /
        \\/ \\/          \\/            \\/      \\/       \\/         \\/
"""
    )
    while True:
        cp.print_magenta("\n--- Menu ---")
        print("1. Extraction de documents de jurisprudence depuis le web")
        print("2. Analyse des documents")
        print("3. Donner des recommendantions")
        print("9. Quitter")

        choice = input("Selectionnez une options (1-9): ")

        if choice == "1":
            retrieve()
        elif choice == "2":
            analyse()
        elif choice == "3":
            recommandations()
        elif choice == "9":
            exit_program()
        else:
            cp.print_red("Invalide")


if __name__ == "__main__":
    main()
