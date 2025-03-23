import colorama
from colorama import Fore, Style
import time

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


def loading_bar(iteration, total):
    decimals = 1
    fill = "█"  # █
    length = 30
    print_end = "\r"
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + " " * (length - filled_length)
    print(f"\r   |{bar}| {percent}%", end=print_end)
    time.sleep(0.2)
