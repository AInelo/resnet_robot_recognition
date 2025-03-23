#!/bin/bash

# Installer les dépendances à partir du fichier requirements.txt
echo "Installation des dépendances..."
pip install -r requirements.txt

# Exécuter le script Python
echo "Exécution de main.py..."
python main.py
