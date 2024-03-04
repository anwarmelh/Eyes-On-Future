# Synapse Analytics notebook source

# METADATA ********************

# META {
# META   "synapse": {
# META     "lakehouse": {
# META       "default_lakehouse": "9732e201-a3a7-49e5-bc5f-2fe15e8c54ab",
# META       "default_lakehouse_name": "LK_weather_raw",
# META       "default_lakehouse_workspace_id": "0e9e1e73-833b-4f95-ac48-74db3d1ccde0",
# META       "known_lakehouses": [
# META         {
# META           "id": "9732e201-a3a7-49e5-bc5f-2fe15e8c54ab"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# # **Load the climate data for British Columbia**

# MARKDOWN ********************

# ## Find station ID

# CELL ********************

df_station = spark.sql("SELECT StationID FROM LK_weather_raw.station_weather_ca where Province='BRITISH COLUMBIA'")

# MARKDOWN ********************

# ## Obtain the data for the desired interval

# CELL ********************

import requests
import csv
import json
from io import StringIO



# Listes des valeurs pour chaque paramètre
stationIDs = [row['StationID'] for row in df_station.collect()] # Exemple d'ID de stations
years = [year for year in range(2003,2023)] # Exemple d'années


# L'URL de base
base_url = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"

# Itérer à travers chaque combinaison de stationID, Year, Month et Day
for stationID in stationIDs:
    for year in years:
        # Générer l'URL
        url = f"{base_url}?format=csv&stationID={stationID}&Year={year}&timeframe=2"
             
        # Ici, vous pouvez ajouter le code pour exécuter l'URL avec, par exemple, requests.get(url) en Python
        # Mais n'oubliez pas de gérer les réponses et les erreurs potentielles.
        response = requests.get(url)
         
        if response.status_code == 200:
            # Décoder les données binaires en texte
            content = response.content.decode('utf-8')
                    
            # Utiliser StringIO pour simuler un fichier en mémoire
            f = StringIO(content)
                    
            # Lire le contenu CSV dans un dictionnaire
            reader = csv.DictReader(f)
            rows = list(reader) # Convertir les lignes CSV en une liste de dictionnaires
                    
            # Convertir la liste de dictionnaires en JSON
            json_data = json.dumps(rows)
                    
            # À ce stade, `json_data` est une chaîne de caractères formatée en JSON
            json_rdd=spark.sparkContext.parallelize([json_data])
            json_df=spark.read.json(json_rdd)
            json_df.write.partitionBy("Year", "Month").mode("append").parquet("Files/Canada/BC/CBCdata.parquet")
        else:
            print(f"Erreur lors de la récupération des données de l'URL {url}")

# MARKDOWN ********************

# ## Optimize delta tables

# CELL ********************

spark.conf.set("spark.sql.parquet.vorder.enabled", "true")
spark.conf.set("spark.microsoft.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.microsoft.delta.optimizeWrite.binSize", "1073741824")

# MARKDOWN ********************

# ## Save the data in a delta table.

# CELL ********************

from notebookutils import mssparkutils
import re



# df now is a Spark DataFrame containing parquet data from "Files/Canada/BC/CBCdata.parquet/Year=2003/Month=01/part-00015-00a3ea7d-f2d6-4470-9682-c48327a19051.c000.snappy.parquet".
 
# Remplacer 'chemin_du_dossier_principal' par le chemin d'accès au dossier principal

years = [year for year in range(2003,2024)] # Exemple d'années
months = [month for month in range(1,13)] # Exemple de mois

for year in years:
        for month in months:
            if month <10 :
                sous_dossiers =f"Files/Canada/BC/CBCdata.parquet/Year={year}/Month=0{month}"
            else:
                sous_dossiers =f"Files/Canada/BC/CBCdata.parquet/Year={year}/Month={month}"
            print(sous_dossiers)
                # Lire les fichiers Parquet du sous-dossier
            df = spark.read.parquet(sous_dossiers)
            def clean_column_name(column_name):
                # Supprime tous les caractères non alphanumériques sauf le underscore
                clean_name = re.sub(r'[^\w_]', '', column_name)
                return clean_name

            # Création d'un dictionnaire pour le renommage
            rename_dict = {col: clean_column_name(col) for col in df.columns}

            # Renommage des colonnes
            df_cleaned = df.selectExpr(*(f"`{col}` as `{rename_dict[col]}`" for col in df.columns))

            # Maintenant, df_cleaned contient les colonnes avec les noms nettoyés


                # Écrire/ajouter le DataFrame dans une table Delta
            df_cleaned.write.format("delta").mode("append").saveAsTable("Weather_Canada_BC")
