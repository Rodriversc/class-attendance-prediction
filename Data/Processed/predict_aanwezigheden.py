"""
Voorspel de aanwezigheidsgraad voor een lijst lessen.

Gebruik:
    python predict_aanwezigheden.py lessen-subgroepen.csv

Het invoerbestand gebruikt ';' als scheidingsteken en bevat:

    date
    from
    until
    activity
    subgroup
    program
    floor
    roomcategory
    capacity
    area
    credits

De oorspronkelijke volgorde van de rijen blijft behouden.

Uitvoer:
    voorspellingen.csv
"""

import sys
import argparse
import numpy as np
import pandas as pd
import joblib


# ============================================================
# INSTELLINGEN
# ============================================================

MODEL_PAD = "model_aanwezigheden.pkl"
OUTPUT_PAD = "voorspellingen.csv"


# ============================================================
# HULPFUNCTIES
# ============================================================

def parse_tijd(t):
    """
    Zet een tijd zoals 815, 0815, 1030 of 10:30 om
    naar (uur, minuut).
    """

    if pd.isna(t):
        return np.nan, np.nan

    tekst = str(t).strip()

    # Voor formaat 08:15 of 08:15:00
    if ":" in tekst:
        delen = tekst.split(":")
        return int(delen[0]), int(delen[1])

    # Voor formaat 815 / 0815 / 1030
    tekst = str(int(float(tekst))).zfill(4)

    return int(tekst[:2]), int(tekst[2:])


def maak_dagdeel(uur):

    if uur < 12:
        return "Ochtend"

    elif uur < 17:
        return "Namiddag"

    else:
        return "Avond"


def maak_groepsgrootte(aantal):

    if aantal <= 10:
        return "1-10"

    elif aantal <= 25:
        return "11-25"

    elif aantal <= 50:
        return "26-50"

    elif aantal <= 100:
        return "51-100"

    else:
        return "100+"


def seizoen_van_maand(maand):

    if maand in [9, 10, 11]:
        return "Autumn"

    elif maand in [12, 1, 2]:
        return "Winter"

    elif maand in [3, 4, 5]:
        return "Spring"

    else:
        return "Summer"


def meest_logische_classcode(
    row,
    classcode_lookup,
    fallback
):
    """
    Zoek ClassCode op basis van:
    SubgroupCode + ProgramName + ClassCredits + CanonicalActivity.

    Wanneer meerdere ClassCodes in de trainingsdata mogelijk waren,
    bevat de lookup de meest voorkomende ClassCode.
    """

    sleutel = (
        row["SubgroupCode"],
        row["ProgramName"],
        row["ClassCredits"],
        row["CanonicalActivity"]
    )

    return classcode_lookup.get(
        sleutel,
        fallback
    )


def schat_total_students(
    row,
    subgroup_lookup,
    program_lookup,
    fallback
):
    """
    TotalStudents is niet aanwezig in de examen-CSV.

    Prioriteit:
    1. mediaan voor dezelfde subgroep
    2. mediaan voor hetzelfde programma
    3. algemene mediaan
    """

    subgroup = row["SubgroupCode"]
    program = row["ProgramName"]

    if subgroup in subgroup_lookup:
        return subgroup_lookup[subgroup]

    if program in program_lookup:
        return program_lookup[program]

    return fallback


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def prepare_data(data, artifact):

    df = data.copy()

    feature_columns = artifact["feature_columns"]
    semester_start = pd.Timestamp(
        artifact["semester_start"]
    )

    classcode_lookup = artifact["classcode_lookup"]
    total_students_subgroup = artifact[
        "total_students_subgroup"
    ]
    total_students_program = artifact[
        "total_students_program"
    ]
    activity_lookup = artifact["activity_lookup"]
    class_lookup = artifact["class_lookup"]
    fallbacks = artifact["fallbacks"]


    # ========================================================
    # 1. BRONKOLOMMEN MAPPEN
    # ========================================================

    # Datum
    df["FullDate"] = pd.to_datetime(
        df["date"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )

    # Activiteit
    df["CanonicalActivity"] = (
        df["activity"]
        .astype(str)
        .str.strip()
    )

    # Subgroep
    df["SubgroupCode"] = (
        df["subgroup"]
        .astype(str)
        .str.strip()
    )

    # Programma
    df["ProgramName"] = (
        df["program"]
        .astype(str)
        .str.strip()
    )

    # Lokaal
    df["RoomFloor"] = pd.to_numeric(
        df["floor"],
        errors="coerce"
    )

    df["RoomCategory"] = (
        df["roomcategory"]
        .astype(str)
        .str.strip()
    )

    df["Capacity"] = pd.to_numeric(
        df["capacity"],
        errors="coerce"
    )

    # Ondersteunt zowel 128.35 als 128,35
    df["Area"] = (
        df["area"]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )

    df["Area"] = pd.to_numeric(
        df["Area"],
        errors="coerce"
    )

    # Studiepunten
    df["ClassCredits"] = pd.to_numeric(
        df["credits"],
        errors="coerce"
    )


    # ========================================================
    # 2. DATUMFEATURES
    # ========================================================

    df["Weekday"] = (
        ((df["FullDate"].dt.dayofweek + 1) % 7)
        + 1
    )

    # De trainingsdata bevat Engelstalige namen
    dag_namen = {
        1: "Sunday",
        2: "Monday",
        3: "Tuesday",
        4: "Wednesday",
        5: "Thursday",
        6: "Friday",
        7: "Saturday"
    }

    df["NameDayDutch"] = df["Weekday"].map(
        dag_namen
    )

    df["Month"] = df["FullDate"].dt.month

    maand_namen = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December"
    }

    df["NameMonthDutch"] = df["Month"].map(
        maand_namen
    )

    df["Year"] = df["FullDate"].dt.year

    df["DayOfYear"] = (
        df["FullDate"].dt.dayofyear
    )

    df["NumberSemester"] = (
        (df["Month"] > 6).astype(int) + 1
    )

    df["Season"] = (
        df["Month"]
        .apply(seizoen_van_maand)
    )

    df["SemesterWeek"] = (
        (
            df["FullDate"]
            - semester_start
        ).dt.days // 7
    ) + 1


    # ========================================================
    # 3. TIJDFEATURES
    # ========================================================

    start = df["from"].apply(parse_tijd)
    einde = df["until"].apply(parse_tijd)

    df["StartHour"] = [
        waarde[0] for waarde in start
    ]

    df["StartMinutes"] = [
        waarde[1] for waarde in start
    ]

    df["EndHour"] = [
        waarde[0] for waarde in einde
    ]

    df["EndMinutes"] = [
        waarde[1] for waarde in einde
    ]

    df["StartSeconds"] = 0
    df["EndSeconds"] = 0

    df["StartTime"] = (
        df["StartHour"].astype(int)
        .astype(str)
        .str.zfill(2)
        + ":"
        + df["StartMinutes"].astype(int)
        .astype(str)
        .str.zfill(2)
        + ":00"
    )

    df["EndTime"] = (
        df["EndHour"].astype(int)
        .astype(str)
        .str.zfill(2)
        + ":"
        + df["EndMinutes"].astype(int)
        .astype(str)
        .str.zfill(2)
        + ":00"
    )

    df["StartAMPM"] = np.where(
        df["StartHour"] < 12,
        "AM",
        "PM"
    )

    df["EndAMPM"] = np.where(
        df["EndHour"] < 12,
        "AM",
        "PM"
    )


    # ========================================================
    # 4. LESDUUR
    # Exact dezelfde logica als tijdens training
    # ========================================================

    df["Lesduur"] = (
        (
            df["EndHour"] * 60
            + df["EndMinutes"]
        )
        -
        (
            df["StartHour"] * 60
            + df["StartMinutes"]
        )
    ) / 60

    df["Lesduur"] = (
        df["Lesduur"] * 2
    ).round() / 2

    df["Dagdeel"] = (
        df["StartHour"]
        .apply(maak_dagdeel)
    )


    # ========================================================
    # 5. TOTALSTUDENTS RECONSTRUEREN
    # ========================================================

    fallback_total = fallbacks.get(
        "TotalStudents",
        20
    )

    df["TotalStudents"] = df.apply(
        lambda row: schat_total_students(
            row,
            total_students_subgroup,
            total_students_program,
            fallback_total
        ),
        axis=1
    )

    df["TotalStudents"] = (
        pd.to_numeric(
            df["TotalStudents"],
            errors="coerce"
        )
        .fillna(fallback_total)
        .round()
        .astype(int)
    )

    df["Groepsgrootte"] = (
        df["TotalStudents"]
        .apply(maak_groepsgrootte)
    )


    # ========================================================
    # 6. ACTIVITEITSFEATURES VIA LOOKUP
    # ========================================================

    activity_features = [
        "ActivityDomain",
        "ActivityGroup",
        "SourceColumn",
        "SourceValue",
        "IsCourse",
        "IsExam",
        "IsAcademicEvent",
        "IsGuestLecture",
        "IsMakeUp",
        "IsPractical",
        "IsPresentation",
        "IsDigitalPossible"
    ]

    for feature in activity_features:

        df[feature] = df[
            "CanonicalActivity"
        ].apply(
            lambda activiteit:
                activity_lookup
                .get(activiteit, {})
                .get(
                    feature,
                    fallbacks.get(feature)
                )
        )


    # ========================================================
    # 7. CLASSCODE RECONSTRUEREN
    # ========================================================

    fallback_classcode = fallbacks.get(
        "ClassCode",
        0
    )

    df["ClassCode"] = df.apply(
        lambda row: meest_logische_classcode(
            row,
            classcode_lookup,
            fallback_classcode
        ),
        axis=1
    )


    # ========================================================
    # 8. CLASSFEATURES VIA CLASSCODE
    # ========================================================

    class_features = [
        "ClassSecondChance",
        "ClassProgramStage"
    ]

    for feature in class_features:

        df[feature] = df[
            "ClassCode"
        ].apply(
            lambda code:
                class_lookup
                .get(code, {})
                .get(
                    feature,
                    fallbacks.get(feature)
                )
        )


    # ========================================================
    # 9. OVERIGE FEATURES
    # ========================================================

    # Building is constant in onze trainingsdata
    df["Building"] = fallbacks.get(
        "Building",
        2
    )

    df["Stakingsdag"] = fallbacks.get(
        "Stakingsdag",
        0
    )


    # ========================================================
    # 10. WEERFEATURES
    #
    # Deze staan niet in het examenbestand.
    # Daarom gebruiken we de typische waarden uit de training.
    # ========================================================

    weer_features = [
        "GemiddeldeTemperatuur",
        "MaximumTemperatuur",
        "MinimumTemperatuur",
        "TotaleNeerslag",
        "TotaleRegen",
        "MaximaleWindsnelheid",
        "WeerCode",
        "ZonneschijnSeconden",
        "DaglichtSeconden",
        "NeerslagUren"
    ]

    for feature in weer_features:
        df[feature] = fallbacks.get(
            feature,
            0
        )

    df["HeeftGeregend"] = (
        df["TotaleRegen"] > 0
    ).astype(int)


    # ========================================================
    # 11. ONTBREKENDE MODELFEATURES OPVANGEN
    # ========================================================

    for feature in feature_columns:

        if feature not in df.columns:
            df[feature] = fallbacks.get(
                feature,
                0
            )


    # ========================================================
    # 12. EXACT DEZELFDE FEATURES EN VOLGORDE
    # ALS BIJ TRAINING
    # ========================================================

    X = df[
        feature_columns
    ].copy()

    return df, X


# ============================================================
# HOOFDFUNCTIE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Voorspel aanwezigheidsgraad "
            "voor een lijst lessen."
        )
    )

    parser.add_argument(
        "invoer_csv",
        help="Pad naar het invoerbestand"
    )

    args = parser.parse_args()


    # ========================================================
    # MODEL LADEN
    # ========================================================

    print(
        f"Model laden uit {MODEL_PAD} ..."
    )

    artifact = joblib.load(
        MODEL_PAD
    )

    model = artifact["model"]


    # ========================================================
    # CSV LADEN
    # ========================================================

    print(
        f"Invoer lezen uit "
        f"{args.invoer_csv} ..."
    )

    data = pd.read_csv(
        args.invoer_csv,
        sep=";",
        encoding="utf-8-sig"
    )

    # Kolomnamen standaardiseren
    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
    )


    # ========================================================
    # VEREISTE KOLOMMEN CONTROLEREN
    # ========================================================

    vereiste_kolommen = [
        "date",
        "from",
        "until",
        "activity",
        "subgroup",
        "program",
        "floor",
        "roomcategory",
        "capacity",
        "area",
        "credits"
    ]

    ontbrekende_kolommen = [
        kolom
        for kolom in vereiste_kolommen
        if kolom not in data.columns
    ]

    if ontbrekende_kolommen:

        print(
            "[FOUT] Ontbrekende kolommen:"
        )

        print(
            ontbrekende_kolommen
        )

        sys.exit(1)


    print(
        f"Aantal rijen ingelezen: "
        f"{len(data)}"
    )


    # ========================================================
    # FEATURES RECONSTRUEREN
    # ========================================================

    prepared, X = prepare_data(
        data,
        artifact
    )


    # ========================================================
    # VOORSPELLEN MET JOUW BESTE MODEL
    # ========================================================

    print(
        "Voorspellingen berekenen "
        "met beste Random Forest-model ..."
    )

    voorspellingen = model.predict(X)

    # Aanwezigheidsgraad logisch tussen 0 en 1
    voorspellingen = np.clip(
        voorspellingen,
        0,
        1
    )


    # ========================================================
    # RESULTAAT
    #
    # We werken op een kopie van de oorspronkelijke CSV.
    # Er wordt NIET gesorteerd.
    # De oorspronkelijke rijvolgorde blijft dus behouden.
    # ========================================================

    resultaat = data.copy()

    resultaat[
        "voorspelde_aanwezigheidsgraad"
    ] = voorspellingen.round(3)

    resultaat[
        "voorspelde_aanwezigheid_pct"
    ] = (
        voorspellingen * 100
    ).round(1)


    # ========================================================
    # RESULTATEN TONEN
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "VOORSPELLINGEN"
    )

    print(
        "=" * 80
    )

    print()

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        250
    ):

        print(
            resultaat.to_string(
                index=False
            )
        )


    # ========================================================
    # RESULTATEN OPSLAAN
    # ========================================================

    resultaat.to_csv(
        OUTPUT_PAD,
        sep=";",
        index=False,
        encoding="utf-8-sig"
    )

    print()

    print(
        "=" * 80
    )

    print(
        f"Resultaten opgeslagen in: "
        f"{OUTPUT_PAD}"
    )

    print(
        "=" * 80
    )


# ============================================================
# SCRIPT STARTEN
# ============================================================

if __name__ == "__main__":
    main()