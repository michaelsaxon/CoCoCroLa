import os
from http import server
import socketserver

import click

from cococrola.utils.build_html import build_squares

#@click.command(cls=CommandWithConfigFile('../config/generate.yaml'))
@click.command()
@click.option('--input_csv', type=str, default="../benchmark/v0-1/concepts.csv")
@click.option('--fname_base', type=str, default="../results/localization_off_en/SD2")
@click.option('--base_language', type=str, default="en")
@click.option('--socket', type=int, default=8000)
def main(input_csv, fname_base, base_language, socket):
    # get word point as the index of the language code in the first row of the csv
    print("Building index.html!")
    index = open(input_csv, "r").readlines()[0].strip().split(",")
    base_word_point = index.index(base_language.lower())
    page_html_lines = build_squares(input_csv, fname_base, base_word_point)
    open("index.html", "w").write(page_html_lines)
    print("Launching web server!")
    with socketserver.TCPServer(("", socket), server.SimpleHTTPRequestHandler) as httpd:
        try:
            print("Web server running... press Ctrl+C to stop")
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()
            print("Web server stopped, deleing index.html")
            os.remove("index.html")
    

if __name__ == "__main__":
    main()