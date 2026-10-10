# Démarche Data Science pour ce projet de reconnaissance faciale

Le dossier contient **3 personnes — daniel, donald et mark — avec 100 images brutes chacune**, ainsi que leurs 100 versions prétraitées. Cela représente **300 observations originales** : les versions prétraitées sont des transformations des mêmes images, et non des observations supplémentaires indépendantes.


## Application de la démarche

|---|---|
| **Dataset brut : types, distributions, qualité, valeurs manquantes** | Examiner les formats, dimensions, nombres d’images par personne, le flou, la luminosité, les doublons, les fichiers illisibles et les visages non détectés. Les « valeurs manquantes » correspondent surtout aux images ou embeddings inutilisables. |
| **EDA + tests statistiques** | Examiner les distributions de qualité, les images atypiques et les distances entre embeddings d’une même personne et de personnes différentes. Étudier une hypothèse comme « les images floues entraînent davantage d’erreurs », si les mesures nécessaires sont disponibles. Tenir compte de la dépendance entre images d’une même session dans les tests. |
| **Prétraitement + Feature Engineering** | Alignement, recadrage, redimensionnement et extraction des **128 caractéristiques FaceNet**. Toute transformation apprise, comme une PCA ou une standardisation, doit être ajustée uniquement sur l’entraînement de chaque fold. |
| **Modélisation + validation croisée** | Comparer **SVM, Logistic Regression, Random Forest et XGBoost sur les mêmes embeddings**, avec les mêmes partitions. Garder le SVM actuel comme référence ; aucun gain des autres modèles n’est garanti. |
| **Évaluation + interprétabilité** | F1 macro, précision/rappel par personne, matrice de confusion, calibration et analyse visuelle des erreurs. PR-AUC peut être calculée par personne en *one-vs-rest*. SHAP est possible sur les embeddings, mais expliquer une dimension FaceNet ne signifie pas expliquer directement une région du visage. |

## Priorité : fiabiliser la validation

Le script capture les images à intervalles de **0,15 seconde**, puis utilise une validation croisée aléatoire. Des images très similaires d’une même séquence peuvent donc se retrouver dans l’entraînement et la validation, ce qui risque de rendre les scores trop optimistes.

Pour mesurer la reconnaissance des personnes connues dans de nouvelles conditions, collecter **plusieurs sessions par personne**, puis séparer les données par session en conservant les personnes connues dans l’entraînement. Les méthodes de validation par groupes permettent cette séparation : [documentation scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html).

Les augmentations d’une même image doivent rester dans la partition de cette image. Un extracteur FaceNet préentraîné et figé peut produire les embeddings avant la séparation, à condition de ne pas être adapté à l’ensemble des données du projet.

## Points à améliorer dans le code actuel

- `face-train.py` évalue uniquement **l’accuracy**, sans test final indépendant.
- Il limite les folds selon le **nombre de classes** ; pour une validation stratifiée, la contrainte concerne le nombre d’exemples de la classe la moins représentée. Une validation par sessions impose aussi de disposer de suffisamment de groupes.
- Il sauvegarde un seuil « Unknown » de **0,55**, alors que `face-recognition.py` utilise **0,70**. Ce seuil devrait être choisi sur une validation contenant aussi des personnes inconnues, puis évalué sur un test séparé.

Afficher une probabilité SVM ne suffit pas à démontrer que la confiance est fiable : la calibration doit être vérifiée sur des données séparées. Voir la [documentation sur la calibration](https://scikit-learn.org/stable/modules/calibration.html).

Pour la fonction « Unknown », compléter les mesures de classification des personnes connues par le taux d’acceptation erronée des inconnus et le taux de rejet erroné des personnes connues.

## Conclusion

**Ce plan convient pour enrichir le projet.** Avec ce petit dataset actuellement équilibré en nombre de fichiers, les apports les plus utiles seront l’analyse de qualité, la collecte de sessions variées, la comparaison des modèles et l’évaluation des erreurs. Les tests statistiques et SHAP peuvent rester complémentaires.

Cette note décrit les adaptations proposées ; elle ne rapporte pas de nouvelles expériences d’entraînement ni de résultats de performance.
