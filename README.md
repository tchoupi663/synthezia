# Synthezia

## Introduction

Synthezia est un outil avancé d'analyse juridique conçu pour traiter, synthétiser et interpréter des documents de jurisprudence. L'outil automatise l'extraction d'informations pertinentes à partir de décisions de justice, notamment de la Cour de cassation, et propose des recommandations basées sur la similarité entre différentes affaires juridiques et un prompt utilisateur. 

## Installation

### Prérequis

- Python 3.8 ou supérieur (3.12 idéalement)
- Gestionnaire de paquets pip
- Clés API pour Gemini et Judilibre (tuto pour la clé API Judilibre [ici](https://github.com/Cour-de-cassation/judilibre-search))

### Étapes d'installation

1. Cloner le dépôt:
   ```
   git clone https://github.com/yourusername/lawthesia.git
   cd lawthesia
   ```

2. Installer les dépendances:
   ```
   pip install -r requirements.txt
   ```

3. Configurer votre environnement virtuel et créez les dossiers pour les données:
   ```
   mkdir -p raw_data processed_data vectors
   cp .env-example .env
   ```
   Modifiez le fichier `.env` avec vos clés API:
   ```
   GEMINI-API_KEY=votre_clé_gemini_ici
   JUDILIBRE_KEY=votre_clé_judilibre_ici
   ```


## Utilisation

Lawthesia propose deux modes d'utilisation: un mode utilisateur standard et un mode développeur.

### Mode utilisateur

Pour lancer l'application en mode utilisateur:

```
python synthezia.py
```

Ce mode affiche un menu simplifié avec l'option principale:
- **Donner des recommandations**: Permet d'entrer une situation juridique et de recevoir des recommandations basées sur des cas similaires.

### Mode développeur

Pour lancer l'application en mode développeur:

```
python synthezia.py --dev
```
ou
```
python synthezia.py -d
```

Ce mode offre des options supplémentaires:
- **Extraction des 10000 premiers documents de jurisprudence**: Récupère des documents juridiques depuis l'API Judilibre.
- **Analyse des documents extraits**: Traite les documents récupérés et extrait des informations structurées.
- **Donner des recommandations**: Même fonctionnalité que dans le mode utilisateur.

## Explication

Lawthesia fonctionne grâce à plusieurs composants clés:

1. **Module d'extraction de données**: Récupère des documents juridiques depuis l'API Judilibre, avec un focus sur les arrêts de la Cour de cassation de la chambre criminelle.
   
2. **Module d'analyse IA**: Utilise l'API Gemini pour analyser les documents juridiques et en extraire des informations structurées comme:
   - Le type de solution juridique
   - Les parties impliquées
   - Le contexte juridique
   - Les faits de l'affaire
   - Le problème juridique central
   - Les arguments des parties
   - Les motifs de la décision
   - Les articles de loi cités et suggérés
   - Les montants financiers en jeu

3. **Système de vectorisation et de similarité**: Transforme les documents en vecteurs TF-IDF pour permettre la recherche de similarités entre différentes affaires juridiques, en appliquant des coefficients d'importance differents pour chaque element.


## Exemples

### Exemple 1: Recherche de recommandations juridiques

Après avoir lancé le programme et choisi l'option "Donner des recommandations", vous pourrez entrer une description de votre situation juridique:

```
Donnez un résumé de la situation juridique, en incluant les éléments suivants:

Conseils pour rédiger votre demande: Structurez votre recherche en incluant :
    (1) le problème juridique précis,
    (2) les articles de loi applicables,
    (3) les faits-clés (montants, durées, sanctions).
    (4) Utilisez des termes techniques (détention provisoire, infractions aux législations, etc.)

-> Un individu a été accusé d'abus de confiance après avoir utilisé les fonds de son entreprise pour des dépenses personnelles d'un montant de 50000 euros. L'article 314-1 du Code pénal pourrait s'appliquer. Quelles sont les sanctions possibles et existe-t-il une jurisprudence similaire?
```

Le système analysera votre texte, le convertira en vecteur et recherchera des cas similaires dans sa base de données. Les résultats afficheront:
- Le pourcentage de similarité avec chaque document
- Les articles de loi pertinents
- Un lien vers la décision complète

Quelques prompts d'dxamples que vous pouvez utiliser:

```
Qu'est-ce qui caractérise suffisamment les délits de détournement de fonds et d'abus de confiance et de complicité de ce délit?
```

```
Les échanges de mails et le rôle central dans le système de contrats fictifs, caractérisent-ils suffisamment les délits de détournement de fonds publics et d'abus de confiance et de complicité de ce délit?
```

```
Mon client a joué un role centrale dans une operation de détournement de fonds de 2900000 euros. Il y a des preuves sous forme d'échange de mail et de contrats fictifs ou falsifiés. Articles 408 du Code pénal, 591 du Code de procédure pénale, 593 du Code de procédure pénale, 931 du Code civil.
```

### Exemple 2: Extraction et analyse de documents (mode développeur)

En mode développeur, vous pouvez extraire de nouveaux documents juridiques:

1. Sélectionnez l'option "Extraction de documents de jurisprudence"
2. Le système récupérera des documents depuis l'API Judilibre
3. Sélectionnez ensuite "Analyse des documents extraits"
4. Le système traitera ces documents avec l'IA pour en extraire des informations structurées
5. Ces documents seront ensuite disponibles pour la recherche de similarité

## Licence

[MIT](LICENSE)
