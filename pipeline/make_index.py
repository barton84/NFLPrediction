"""Wrap site/app.html into a full standalone site/index.html for Netlify."""
import os
ROOT = os.path.join(os.path.dirname(__file__), "..", "site")
body = open(os.path.join(ROOT, "app.html")).read()
html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<style>body{margin:0}[hidden]{display:none!important}</style>\n</head>\n<body>\n' + body + '\n</body>\n</html>\n')
open(os.path.join(ROOT, "index.html"), "w").write(html)
print("wrote site/index.html")
