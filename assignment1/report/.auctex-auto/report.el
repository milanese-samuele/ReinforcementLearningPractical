(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "a4paper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("babel" "english") ("inputenc" "utf8x") ("todonotes" "colorinlistoftodos") ("natbib" "authoryear") ("geometry" "margin=1in")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "babel"
    "inputenc"
    "amsmath"
    "amssymb"
    "float"
    "graphicx"
    "todonotes"
    "natbib"
    "hyperref"
    "authblk"
    "geometry"))
 :latex)

