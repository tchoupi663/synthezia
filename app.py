#!/usr/bin/env python3

import utils.colour_utils.colour_printer as cp
import utils.fileRead_utils.fileRead_utils as pu
import utils.scrape_utils.scrape_utils as sc


def display_message():
    cp.print_red("You selected Option 1: Display a message.")
    print("This is a temporary message.")


def scrape():
    print("Combien de documents de jurisprudence voulez-vous extraire?")
    num_pages = int(input("Entrez un nombre: "))
    sc.extract_from_website(num_pages)


def exit_program():
    cp.print_blue("You selected Option 3: Exit the program.")
    cp.print_blue("Exiting...")
    exit()


def main():
    while True:
        cp.print_magenta("\n--- Menu ---")
        print("1. Display a message")
        print("2. Extraction de documents de jurisprudence")
        print("3. Exit the program")

        choice = input("Please select an option (1-3): ")

        if choice == "1":
            display_message()
        elif choice == "2":
            scrape()
        elif choice == "3":
            exit_program()
        else:
            cp.print_red("Invalid option. Please select a valid option (1-3).")


if __name__ == "__main__":
    main()
