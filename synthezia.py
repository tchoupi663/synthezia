#!/usr/bin/env python3

import utils.terminal_utils.terminal_utils as cp
import utils.dbUpdate.dbUpdate as db
import utils.ai_utils.ai_utils as ai
from sys import exit, argv
import os


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def display_message():
    cp.print_red("You selected Option 1: Display a message.")
    print("This is a temporary message.")


def retrieve():
    db.retrieve_all()
    input("Appuyez sur entrée pour continuer...")


def analyse():
    ai.summarize_all_files()
    input("Appuyez sur entrée pour continuer...")


def recommandations():
    ai.find_recommendations()
    input("Appuyez sur entrée pour continuer...")


def exit_program():
    cp.print_blue("Quitter...")
    exit()


def devMenu():
    while True:
        clear_screen()
        cp.print_magenta(
            """\n\n
                  _________                __   .__                     .__
                 /   _____/___.__.  ____ _/  |_ |  |__    ____  ________|__|_____
                 \\_____  \\<   |  | /    \\\\   __\\|  |  \\ _/ __ \\ \\___   /|  |\\__  \\
                 /        \\\\___  ||   |  \\\\  |  |   Y  \\\\  ___/  /   _/ |  | / __ \\_
                /_______  // ____||___|  /|__|  |___|  / \\___  >/_____ \\|__|(____  /
                        \\/ \\/          \\/            \\/      \\/       \\/         \\/ {v0.1.0}
                \n
"""
        )

        print("             1. Extraction de documents de jurisprudence")
        print("               2. Analyse des documents extraits")
        cp.print_magenta("                 3. Donner des recommendantions")
        cp.print_blue("                   9. Quitter")
        choice = input("                        Selectionnez une option (1-9): ")

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


def userMenu():
    while True:
        clear_screen()
        cp.print_magenta(
            """\n\n
                  _________                __   .__                     .__
                 /   _____/___.__.  ____ _/  |_ |  |__    ____  ________|__|_____
                 \\_____  \\<   |  | /    \\\\   __\\|  |  \\ _/ __ \\ \\___   /|  |\\__  \\
                 /        \\\\___  ||   |  \\\\  |  |   Y  \\\\  ___/  /   _/ |  | / __ \\_
                /_______  // ____||___|  /|__|  |___|  / \\___  >/_____ \\|__|(____  /
                        \\/ \\/          \\/            \\/      \\/       \\/         \\/ {v0.1.0}
                \n
"""
        )

        cp.print_magenta("                     1. Donner des recommendantions")
        cp.print_blue("                        9. Quitter")
        choice = input("                           Sélectionnez une option (1 ou 9): ")

        if choice == "1":
            recommandations()
        elif choice == "9":
            exit_program()
        else:
            cp.print_red("Invalide")


def main():
    if len(argv) < 2:
        userMenu()
    elif (len(argv) >= 2) and (argv[1] == "--dev") or (argv[1] == "-d"):
        devMenu()


if __name__ == "__main__":
    main()
