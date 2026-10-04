#!/bin/bash
# Retouche standard WineFolio : ./retouche.sh photo.jpg sortie.jpg
# 1) détourage + suppression des ombres sur l'étiquette  2) reconstitution du haut de capsule si coupé
# 3) fond blanc, ombre au sol, format portrait 800x1000
# Dépendances : pip install rembg onnxruntime opencv-python-headless pillow
set -e; D=$(dirname "$0")
python3 "$D/1_detourage_ombres.py" "$1" /tmp/wf_a.png
python3 "$D/2_capsule.py" /tmp/wf_a.png /tmp/wf_b.png
python3 "$D/3_mise_en_page.py" /tmp/wf_b.png "$2"
