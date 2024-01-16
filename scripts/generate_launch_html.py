# use this script in a forwarding localhost ssh like ssh -L 9090:localhost:8000 user@host 
# then you can view the results in your browser at localhost:9090

import os
from http import server
import socketserver

import click

from cococrola.utils.build_html import build_squares, build_columns


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


server.SimpleHTTPRequestHandler.extensions_map = {
    k: v + ';charset=UTF-8' for k, v in server.SimpleHTTPRequestHandler.extensions_map.items()}

class NoCacheHTTPRequestHandler(
    server.SimpleHTTPRequestHandler
):
    def send_response_only(self, code, message=None):
        super().send_response_only(code, message)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-store, must-revalidate')
        self.send_header('Expires', '0')


#@click.command(cls=CommandWithConfigFile('../config/generate.yaml'))
@click.command()
@click.option('--input_csv', type=str, default="../experiments/localization/concepts.csv")
@click.option('--images_dir', type=str, default="../results/localization_off_en/SD2")
@click.option('--base_language', type=str, default="en")
@click.option('--socket', type=int, default=8000)
def main(input_csv, images_dir, base_language, socket):
    print("Building index.html!")
    index = open(input_csv, "r").readlines()[0].strip().split(",")
    base_word_point = index.index(base_language.lower())
    page_html_lines = build_columns(input_csv, "", base_word_point)
    print("CDing to images dir")
    with cd(images_dir):
        # get word point as the index of the language code in the first row of the csv
        print("Saving index.html")
        open("index.html", "w").write(page_html_lines)
        print("Launching web server!")
        with socketserver.TCPServer(("", socket), NoCacheHTTPRequestHandler) as httpd:
            try:
                print("Web server running... press Ctrl+C to stop")
                httpd.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                httpd.shutdown()
                httpd.server_close()
                print("Web server stopped, deleting index.html")
                os.remove("index.html")
    print("Returned to main dir! Exiting")

if __name__ == "__main__":
    main()