"""Wrap site/app.html into a full standalone site/index.html for Netlify."""
import os
ROOT = os.path.join(os.path.dirname(__file__), "..", "site")
body = open(os.path.join(ROOT, "app.html")).read()
html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<link rel="icon" href="favicon.ico" sizes="any">\n'
        '<link rel="icon" type="image/png" sizes="32x32" href="favicon-32.png">\n'
        '<link rel="apple-touch-icon" href="apple-touch-icon.png">\n'
        '<meta name="theme-color" content="#0b1730">\n'
        '<style>body{margin:0}[hidden]{display:none!important}</style>\n</head>\n<body>\n' + body + '\n</body>\n</html>\n')
open(os.path.join(ROOT, "index.html"), "w").write(html)
print("wrote site/index.html")
