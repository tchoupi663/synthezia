"""
This module provides utility functions for printing colored text to the console
using the `colorama` library. Each function corresponds to a specific color and
automatically resets the style after printing.

Functions:
    print_red(text: str) -> None:
        Prints the given text in red color.

    print_green(text: str) -> None:
        Prints the given text in green color.

    print_blue(text: str) -> None:
        Prints the given text in blue color.

    print_yellow(text: str) -> None:
        Prints the given text in yellow color.

    print_cyan(text: str) -> None:
        Prints the given text in cyan color.

    print_magenta(text: str) -> None:
        Prints the given text in magenta color.

    print_white(text: str) -> None:
        Prints the given text in white color.

    print_black(text: str) -> None:
        Prints the given text in black color.
"""

import colorama
from colorama import Fore, Style

colorama.init()


def print_red(text):
    print(Fore.RED + text + Style.RESET_ALL)


def print_green(text):
    print(Fore.GREEN + text + Style.RESET_ALL)


def print_blue(text):
    print(Fore.BLUE + text + Style.RESET_ALL)


def print_yellow(text):
    print(Fore.YELLOW + text + Style.RESET_ALL)


def print_cyan(text):
    print(Fore.CYAN + text + Style.RESET_ALL)


def print_magenta(text):
    print(Fore.MAGENTA + text + Style.RESET_ALL)


def print_white(text):
    print(Fore.WHITE + text + Style.RESET_ALL)


def print_black(text):
    print(Fore.BLACK + text + Style.RESET_ALL)
