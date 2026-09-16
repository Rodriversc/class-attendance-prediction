# Voorspellen van lesaanwezigheid met Machine Learning

Machine-learningproject voor het voorspellen van de **aanwezigheidsgraad tijdens lessen** op basis van historische les-, studenten-, lokaal-, tijds- en weergegevens.

De brondata werd vanuit een SQL-database geëxporteerd naar CSV. In de notebook wordt deze dataset verder verrijkt met historische weerdata van de Open-Meteo API.

## Doel van het project

Het doel is om de toekomstige aanwezigheid tijdens lessen te voorspellen.

De doelvariabele is de **aanwezigheidsgraad**:

```text
aanwezigheidsgraad = UserCount / TotalStudents
```

`UserCount` is gebaseerd op het aantal waargenomen wifi-verbindingen tijdens een lesmoment. Om target leakage te vermijden wordt `UserCount` niet als inputfeature voor het model gebruikt.

## Projectworkflow

De notebook doorloopt de volgende stappen:

1. Data importeren uit een CSV-export van SQL
2. Historische weerdata ophalen via de Open-Meteo API
3. Beide databronnen samenvoegen
4. Exploratory Data Analysis (EDA)
5. Data cleaning
6. Feature engineering
7. Train- en testdata voorbereiden
8. Meerdere regressiemodellen vergelijken
9. Time-series cross-validation uitvoeren
10. Hyperparameters optimaliseren
11. Het definitieve model evalueren op een aparte testset
12. Het getrainde model en de nodige lookup-tabellen opslaan

## Modellen

Onder andere de volgende modellen werden onderzocht:

- Dummy Regressor (baseline)
- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

Voor Random Forest en XGBoost werd bijkomende hyperparameteroptimalisatie uitgevoerd met `RandomizedSearchCV` en `TimeSeriesSplit`.

## Definitief model

Hoewel XGBoost tijdens de time-series cross-validation een iets lagere MAE behaalde, presteerde **Random Forest** beter op de afzonderlijke testset.

Resultaten van het definitieve Random Forest-model:

| Metriek | Resultaat |
|---|---:|
| MAE | 0.1570 |
| RMSE | 0.2072 |
| R² | 0.3166 |

Een MAE van `0.1570` betekent dat de voorspelde aanwezigheidsgraad gemiddeld ongeveer **15,7 procentpunt** afwijkt van de werkelijke aanwezigheidsgraad.

## Belangrijkste features

Volgens de feature importance van het Random Forest-model behoren onder andere deze kenmerken tot de belangrijkste voorspellers:

- `ClassCode`
- `TotalStudents`
- `ClassCredits`
- lokaaloppervlakte (`Area`)
- lokaalcapaciteit (`Capacity`)
- lokaaltype
- tijdskenmerken
- weerkenmerken

De voorspelling wordt dus bepaald door een combinatie van les-, lokaal-, tijds- en weersinformatie.

## Feature engineering

Tijdens het project werden verschillende nieuwe features onderzocht en/of aangemaakt, waaronder:

- semesterweek
- lesduur
- groepsgrootte
- dagdeel
- regenindicator

Niet alle onderzochte features werden behouden. Zo werd een afzonderlijke weekendfeature niet gebruikt omdat weekendlessen nauwelijks voorkwamen en deze informatie al vervat zat in de weekdag. Ook de lokaalbezettingsgraad werd niet behouden omdat deze in bepaalde situaties misleidend kon zijn.

## Repositorystructuur

```text
class-attendance-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
├── ML_Project_RV_finaal.ipynb
│
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
│
├── models/
│   └── README.md
│
├── outputs/
│   ├── figures/
│   └── results/
│
├── sql/
│   └── README.md
│
└── docs/
```

## Data

De notebook verwacht momenteel het bestand:

```text
dataset_aanwezigheden.csv
```

in dezelfde map als de notebook.

De dataset is afkomstig uit een SQL-export en wordt **niet in deze publieke repository opgenomen**, omdat de brondata mogelijk interne of gevoelige onderwijsgegevens bevat.

Voor lokaal gebruik kan het CSV-bestand naast de notebook worden geplaatst.

De map `data/` is voorzien om later een duidelijkere scheiding tussen ruwe en verwerkte data te maken.

## SQL

De map `sql/` kan gebruikt worden om de SQL-query of queries te bewaren waarmee `dataset_aanwezigheden.csv` werd samengesteld.

Wanneer de SQL-code geen gevoelige informatie bevat, is het nuttig om deze mee te publiceren. Zo wordt duidelijk hoe de brondata voor het ML-project werd opgebouwd.

## Opgeslagen model

Na training wordt het definitieve model opgeslagen als:

```text
model_aanwezigheden.pkl
```

Het bestand bevat niet alleen het getrainde model, maar ook:

- de verwachte featurekolommen
- semesterinformatie
- lookup-tabellen
- fallbackwaarden

Hierdoor kan het model later nieuwe data op een consistente manier verwerken.

## Installatie

Maak bij voorkeur een virtuele Python-omgeving aan en installeer vervolgens de dependencies:

```bash
pip install -r requirements.txt
```

Start daarna Jupyter Notebook of open het notebookbestand in VS Code.

## Technologieën

- Python
- Jupyter Notebook
- pandas
- NumPy
- matplotlib
- seaborn
- scikit-learn
- XGBoost
- SciPy
- joblib
- Requests
- Open-Meteo API

## Opmerking

Dit project werd ontwikkeld als machine-learningproject rond het voorspellen van lesaanwezigheid. De resultaten tonen dat historische les- en contextinformatie voorspellende waarde bevat, maar dat een aanzienlijk deel van de variatie in aanwezigheid niet door de beschikbare features kan worden verklaard.
