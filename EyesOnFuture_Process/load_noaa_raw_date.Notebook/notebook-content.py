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

# # Chargement des données NOAA

# CELL ********************

spark.conf.set("spark.sql.parquet.vorder.enabled", "true")
spark.conf.set("spark.microsoft.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.microsoft.delta.optimizeWrite.binSize", "1073741824")

# MARKDOWN ********************

# Initiation de l'API

# CELL ********************

import requests

# Votre token d'API
api_token = 'CxqlSHcxCZwKPKfaPAYIyHiuerpiaqHx'

# URL de base de l'API NOAA
base_url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/'

# Exemple d'endpoint : données d'anomalies de température globales
endpoint = 'datacategories?datasetid=GHCND' # Assurez-vous de remplacer ceci par l'endpoint correspondant à votre requête

# En-têtes avec votre token d'API
headers = {
    'token': api_token
}



# MARKDOWN ********************

# 2 - Appeler l'API

# CELL ********************

# Effectuer la requête
response = requests.get(base_url + endpoint, headers=headers)

# Vérifier le statut de la réponse
if response.status_code == 200:
    # Convertir la réponse en JSON
    data = response.json()
    print(data)
else:
    print(f"Erreur lors de la requête : {response.status_code}")


# MARKDOWN ********************

# 3 - Identifier les sources

# CELL ********************

endpoint = 'datasets'
response = requests.get(f"{base_url}{endpoint}", headers=headers)
if response.status_code == 200:
    datasets = response.json()
    for dataset in datasets['results']:
        print(dataset['id'],' | ', dataset['name'])

# MARKDOWN ********************

# 4- Explorer les données

# CELL ********************

params = {'datasetid': 'GHCND'} # Par exemple pour GHCN Daily
response = requests.get(f"{base_url}datacategories", headers=headers, params=params)
if response.status_code == 200:
    categories = response.json()
    for category in categories['results']:
        print(category['id'],' | ', category['name'])


# MARKDOWN ********************

# 5- Selectionner les types de données

# CELL ********************



params = {'datasetid': 'GHCND', 'datacategoryid': 'TEMP'} # TEMP pour les températures
response = requests.get(f"{base_url}datatypes", headers=headers, params=params)
if response.status_code == 200:
    types = response.json()
    for datatype in types['results']:
        print(datatype['id']," | ", datatype['name'])



# MARKDOWN ********************

# 6- Trouver les stations de mesures

# CELL ********************

params = {'datasetid': 'GHCND', 'datatypeid': 'TMAX', 'limit': 1000} # TMAX pour la température maximale quotidienne
response = requests.get(f"{base_url}stations", headers=headers, params=params)
if response.status_code == 200:
    stations = response.json()
    for station in stations['results']:
        print(station['id'], station['name'])


# MARKDOWN ********************

# 7- Afficher les données

# CELL ********************

params = {
    'datasetid': 'GHCND',
    'datatypeid': 'TMAX',
    'stationid': 'GHCND:USW00094728', # Exemple de station
    'startdate': '2023-01-01',
    'enddate': '2023-01-31',
    'limit': 1000
}
response = requests.get(f"{base_url}data", headers=headers, params=params)
if response.status_code == 200:
    temp_data = response.json()
    for entry in temp_data['results']:
        print(entry['date'], entry['value'])



# MARKDOWN ********************

# try us xx

# CELL ********************

params = {
    'datasetid': 'GHCND',
    'datatypeid': 'TMAX',
    'locationid': 'FIPS:US', # Localisation
    'startdate': '2023-01-01',
    'enddate': '2023-01-31',
    'limit': 1000
}
response = requests.get(f"{base_url}data", headers=headers, params=params)
if response.status_code == 200:
    temp_data = response.json()
    for entry in temp_data['results']:
        print(entry['date'], entry['value'])





# CELL ********************

import pyspark.sql.functions as f
from datetime import datetime

# Supposons que base_url et headers soient déjà définis

# Boucle sur chaque année sur une période de 20 ans
for year in range(2023, 2025): # Ajustez selon la période souhaitée
    # Mettre à jour les paramètres pour l'année courante
    params['startdate'] = f'{year}-01-01'
    params['enddate'] = f'{year}-01-31'
    params['offset'] = 1 # Réinitialiser l'offset pour chaque nouvelle année

    more_pages = True
    while more_pages:
        # Faire la requête GET
        response = requests.get(f"{base_url}data", headers=headers, params=params)

        if response.status_code == 200:
            temp_data = response.json()
            
            # Vérifier s'il y a des résultats à traiter
            if 'results' in temp_data and temp_data['results']:
                # Transformer la réponse en DataFrame PySpark
                df = spark.createDataFrame(temp_data['results'])
                df = df.withColumn('years', f.year(f.to_date('date'))).withColumn('months', f.month(f.to_date('date')))

                # Enregistrer le DataFrame au format Parquet dans le Data Lake
                df.write.partitionBy("years").mode("append").parquet("Files/TMAX/US/US-TMAX.parquet")

                # Mettre à jour l'offset pour la page suivante
                params['offset'] += params['limit']

                # Vérifier si nous avons atteint la fin des résultats
                if len(temp_data['results']) < params['limit']:
                    more_pages = False
            else:
                more_pages = False
        else:
            print(f"Failed to retrieve data for {year}: {response.status_code}")
            more_pages = False


# MARKDOWN ********************

# Données Canada

# CELL ********************

params = {
    'datasetid': 'GHCND',
    'datatypeid': 'TMAX',
    'locationid': 'FIPS:CA', # Localisation
    'startdate': '2023-01-01',
    'enddate': '2023-01-31',
    'limit': 1000
}
response = requests.get(f"{base_url}data", headers=headers, params=params)
if response.status_code == 200:
    temp_data = response.json()
    for entry in temp_data['results']:
        print(entry['date'], entry['value'])





# CELL ********************

import pyspark.sql.functions as f
from datetime import datetime

# Supposons que base_url et headers soient déjà définis

# Boucle sur chaque année sur une période de 20 ans
for year in range(2024, 2025): # Ajustez selon la période souhaitée
    # Mettre à jour les paramètres pour l'année courante
    params['startdate'] = f'{year}-01-01'
    params['enddate'] = f'{year}-01-31'
    params['offset'] = 1 # Réinitialiser l'offset pour chaque nouvelle année

    more_pages = True
    while more_pages:
        # Faire la requête GET
        response = requests.get(f"{base_url}data", headers=headers, params=params)

        if response.status_code == 200:
            temp_data = response.json()
            
            # Vérifier s'il y a des résultats à traiter
            if 'results' in temp_data and temp_data['results']:
                # Transformer la réponse en DataFrame PySpark
                df = spark.createDataFrame(temp_data['results'])
                df = df.withColumn('years', f.year(f.to_date('date'))).withColumn('months', f.month(f.to_date('date')))

                # Enregistrer le DataFrame au format Parquet dans le Data Lake
                df.write.partitionBy("years").mode("append").parquet("Files/TMAX/Canada/Canada-TMAX.parquet")

                # Mettre à jour l'offset pour la page suivante
                params['offset'] += params['limit']

                # Vérifier si nous avons atteint la fin des résultats
                if len(temp_data['results']) < params['limit']:
                    more_pages = False
            else:
                more_pages = False
        else:
            print(f"Failed to retrieve data for {year}: {response.status_code}")
            more_pages = False


# MARKDOWN ********************

# Données Mexique

# CELL ********************

params = {
    'datasetid': 'GHCND',
    'datatypeid': 'TMAX',
    'locationid': 'FIPS:MX', # Localisation
    'startdate': '2023-01-01',
    'enddate': '2023-01-31',
    'limit': 1000
}
response = requests.get(f"{base_url}data", headers=headers, params=params)
if response.status_code == 200:
    temp_data = response.json()
    for entry in temp_data['results']:
        print(entry['date'], entry['value'])





# CELL ********************

import pyspark.sql.functions as f
from datetime import datetime

# Supposons que base_url et headers soient déjà définis

# Boucle sur chaque année sur une période de 20 ans
for year in range(1994, 2025): # Ajustez selon la période souhaitée
    # Mettre à jour les paramètres pour l'année courante
    params['startdate'] = f'{year}-01-01'
    params['enddate'] = f'{year}-01-31'
    params['offset'] = 1 # Réinitialiser l'offset pour chaque nouvelle année

    more_pages = True
    while more_pages:
        # Faire la requête GET
        response = requests.get(f"{base_url}data", headers=headers, params=params)

        if response.status_code == 200:
            temp_data = response.json()
            
            # Vérifier s'il y a des résultats à traiter
            if 'results' in temp_data and temp_data['results']:
                # Transformer la réponse en DataFrame PySpark
                df = spark.createDataFrame(temp_data['results'])
                df = df.withColumn('years', f.year(f.to_date('date'))).withColumn('months', f.month(f.to_date('date')))

                # Enregistrer le DataFrame au format Parquet dans le Data Lake
                df.write.partitionBy("years").mode("append").parquet("Files/TMAX/Mexico/Mexico-TMAX.parquet")

                # Mettre à jour l'offset pour la page suivante
                params['offset'] += params['limit']

                # Vérifier si nous avons atteint la fin des résultats
                if len(temp_data['results']) < params['limit']:
                    more_pages = False
            else:
                more_pages = False
        else:
            print(f"Failed to retrieve data for {year}: {response.status_code}")
            more_pages = False


# MARKDOWN ********************

# Charger les stations

# CELL ********************

params = {'datasetid': 'GHCND', 'datatypeid': 'TMAX', 'limit': 1000} # TMAX pour la température maximale quotidienne
response = requests.get(f"{base_url}stations", headers=headers, params=params)
if response.status_code == 200:
    stations = response.json()
    for station in stations['results']:
        display(stations)


# CELL ********************

import pyspark.sql.functions as f
from datetime import datetime
from pyspark.sql.types import StructType, StructField, StringType

schema = StructType([
    StructField("elevation", StringType(), True),
    StructField("mindate", StringType(), True),
    StructField("maxdate", StringType(), True),
    StructField("latitude", StringType(), True),
    StructField("name", StringType(), True), 
    StructField("datacoverage", StringType(), True),
    StructField("id", StringType(), True),
    StructField("elevationUnit", StringType(), True),
    StructField("longitude", StringType(), True)
])

# Supposons que base_url et headers soient déjà définis

# Boucle sur chaque année sur une période de 20 ans
params['offset'] = 1 # Réinitialiser l'offset pour chaque nouvelle année

more_pages = True
while more_pages:
    # Faire la requête GET
    response = requests.get(f"{base_url}stations", headers=headers, params=params)

    if response.status_code == 200:
        temp_data = response.json()
            
        # Vérifier s'il y a des résultats à traiter
        if 'results' in temp_data and temp_data['results']:
            # Transformer la réponse en DataFrame PySpark
            df = spark.createDataFrame(temp_data['results'],schema)

            # Enregistrer le DataFrame au format Parquet dans le Data Lake
            df.write.mode("append").parquet("Files/Stations/Stations.parquet")

            # Mettre à jour l'offset pour la page suivante
            params['offset'] += params['limit']

            # Vérifier si nous avons atteint la fin des résultats
            if len(temp_data['results']) < params['limit']:
                more_pages = False
        else:
            more_pages = False
    else:
        print(f"Failed to retrieve data : {response.status_code}")
        more_pages = False


# CELL ********************

df = spark.read.parquet("Files/Stations/Stations.parquet/part-00003-4e625967-f74b-4ac0-8969-c0c7205dfec8-c000.snappy.parquet")
# df now is a Spark DataFrame containing parquet data from "Files/Stations/Stations.parquet/part-00003-4e625967-f74b-4ac0-8969-c0c7205dfec8-c000.snappy.parquet".
display(df)
