# Synapse Analytics notebook source

# METADATA ********************

# META {
# META   "synapse": {
# META     "lakehouse": {
# META       "default_lakehouse": "b5ac2035-c35c-42c8-95a7-9bcdcf340977",
# META       "default_lakehouse_name": "LW_weather_analysis",
# META       "default_lakehouse_workspace_id": "0e9e1e73-833b-4f95-ac48-74db3d1ccde0"
# META     },
# META     "environment": {
# META       "environmentId": "cf2eb93c-e6c0-408b-822b-1a52dfa7fe58",
# META       "workspaceId": "79b52fa3-7500-485b-b008-8bdaeac700d7"
# META     }
# META   }
# META }

# MARKDOWN ********************

# # **Loading live data for Vicotia, CA**

# MARKDOWN ********************

# 1. **Establish Normal Temperature Threshold**: Determine a baseline for normal temperatures to identify deviations that may indicate anomalies.
# 
# 2. **Collect Weather Data**: Fetch weather data using the OpenWeatherMap API and store this information in Parquet files for efficient data management.
# 
# 3. **Load Data into KQL Database**: Import the live weather data into a KQL (Kusto Query Language) database for real-time analysis and query capabilities.
# 
# 4. **Anomaly Detection and Analysis**: In instances where current temperatures diverge significantly from the norm (indicating potential anomalies), leverage ChatGPT to conduct a comparative analysis against weather data from the past decade to extract meaningful insights.
# 
# 5. **Respond to Anomalies**: Should an anomaly be confirmed, trigger a pipeline (with ExitValue=1) to acquire satellite imagery from NASA, specifically focusing on global thermal anomalies, to further investigate and understand the detected irregularities.

# MARKDOWN ********************

# ## Define the normal temperature threshold to identify anomalies.

# CELL ********************

AnomalieTemp=5

# MARKDOWN ********************

#  ## Import library

# CELL ********************

from pyspark.sql.functions import year, month, dayofmonth, current_date
from pyspark.sql.functions import to_json, struct, collect_list, concat_ws, lit
from pyspark.sql import SparkSession
import requests
from datetime import datetime
import openai
from openai import AzureOpenAI
from pyspark.sql.functions import lit, current_timestamp
from decimal import Decimal
from notebookutils import mssparkutils
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DecimalType

# MARKDOWN ********************

# ## Optimize delta tables

# CELL ********************

spark.conf.set("spark.sql.parquet.vorder.enabled", "true")
spark.conf.set("spark.microsoft.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.microsoft.delta.optimizeWrite.binSize", "1073741824")

# MARKDOWN ********************

# ## Get weather data from Openweathermap API and save it as parquets files

# CELL ********************


# Paramètres de l'API OpenWeatherMap
api_key = "0c8ee0956c2778f54bd47ed112319f4b" # Remplacez ceci par votre clé API
city_name = "Victoria,CA" # Victoria en Colombie-Britannique, Canada
url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"

# Faire une requête API
response = requests.get(url)
data = response.json()

# Capturer la date et l'heure actuelles comme la date de l'observation
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

try:
    rain_volume = data["rain"]["1h"]
except:
    rain_volume=0

# Préparer les données pour PySpark
weather_data = {
    "City": data["name"],
    "Temperature": data["main"]["temp"],
    "TemperatureMin": data["main"]["temp_min"],
    "TemperatureMax": data["main"]["temp_max"],
    "Humidity": data["main"]["humidity"],
    "Pressure": data["main"]["pressure"],
    "Description": data["weather"][0]["description"],
    "ObservationTime":current_time,
    "Rain":rain_volume
}


# Créer un DataFrame PySpark
df = spark.createDataFrame([weather_data])


# Afficher le DataFrame

df.write.mode("append").parquet("Files/LiveData/")

# MARKDOWN ********************

# ## Select needed data

# CELL ********************

df_sort=df.select("City","Temperature","TemperatureMin","TemperatureMax","Humidity","Pressure","Description","ObservationTime","Rain")

# MARKDOWN ********************

# ## Load live data in KQL database 

# CELL ********************

# Example of query for reading data from Kusto. Replace T with your <tablename>.
kustoQuery = "['WeatherData'] | take 10"
kustoTable="WeatherData"
# The query URI for reading the data e.g. https://<>.kusto.data.microsoft.com.
kustoUri = "https://trd-uh5qsaha2jcsv3zrmv.z9.kusto.fabric.microsoft.com"
# The database with data to be read.
database = "KQL_DB_live_weather"
# The access credentials.
accessToken = mssparkutils.credentials.getToken(kustoUri)
df_sort.write\
    .format("com.microsoft.kusto.spark.synapse.datasource")\
    .option("accessToken", accessToken)\
    .option("kustoCluster", kustoUri)\
    .option("kustoDatabase", database)\
    .option("kustoQuery", kustoQuery)\
    .option("kustoTable", kustoTable)\
    .mode("append").save()

# MARKDOWN ********************

# ## Get current temperature 

# CELL ********************

checkCurrentTemp=(df_sort.select('Temperature'))

# MARKDOWN ********************

# ## Initialise AzureIA CHATGPT

# CELL ********************

api_key= '41e0a22e-2816-4a92-b624-5a833dd0a6b6'
url="https://polite-ground-030dc3103.4.azurestaticapps.net/api/v1"

client = AzureOpenAI(
    azure_endpoint=url,
    api_key=api_key,
    api_version="2023-09-01-preview",
)

MESSAGES = [
    {"role": "system", "content": """You are expert scientist in Meteology"""}
    ]




def callchatgpt(checkrow):
    MESSAGES.append({"role": "user", "content": "In the context of prevention against cliatic risks, I will give you the weather data of the day as well as the data of the last 10 years. Analyze and compare the present with the past and give me a complete summary of what can be understood from it. If you're missing any info, just ignore the point in question and don't mention it. also makes a remark about global warming. I want a clear and precise and short answer in less than 4000 tokens : "+checkrow})
    completion = client.chat.completions.create(model="gpt-4", messages=MESSAGES,temperature=0.2)
    return completion.choices[0].message.content

# MARKDOWN ********************

# ## If the current temperature represent anomalies use ChatGPT to compare it with our last 10 years data and give Insights

# CELL ********************

currentTemp=checkCurrentTemp.collect()[0]['Temperature']
if currentTemp >=AnomalieTemp :
    
    # Convertissez les colonnes Datetime, Totalprecipmm, et MeanTempC en JSON
    df_json_current = df_sort.select(to_json(struct("ObservationTime", "Rain", "Temperature")).alias("data_as_json"))
    text_json_current=df_json_current.collect()[0]['data_as_json']
    

    # Charger la table WeatherData
    # Assurez-vous que le chemin d'accès et le format sont corrects pour votre environnement spécifique
    df = spark.sql("SELECT * FROM LW_weather_analysis.WeatherHistory ")

    # Filtrer pour les enregistrements de "EQUIMALT HARBOUR" des 10 dernières années pour le même jour d'aujourd'hui
    df_filtered = df.filter(\
        (month(df["Datetime"]) == month(current_date())) &\
        (dayofmonth(df["Datetime"]) == dayofmonth(current_date()))\
    ).filter(\
        year(df["Datetime"]) >= year(current_date()) - 10\
    )

    # Convertissez chaque ligne en JSON
    df_json = df_filtered.select(to_json(struct(*df.columns)).alias("json"))

    # Agrégez toutes les lignes JSON en une liste, puis concaténez-les en une seule chaîne JSON
    df_json_aggregated = df_json.agg(concat_ws(", ", collect_list("json")).alias("json_array"))

    # Pour créer un objet JSON valide, vous pouvez ajouter des crochets autour de la chaîne
    df_json_history = df_json_aggregated.select(concat_ws("", lit("["), "json_array", lit("]")).alias("single_json_line"))
    text_json_history=df_json_history.collect()[0]['single_json_line']
    try :
        analyisGPT=callchatgpt(f"Current : {text_json_current}, History : {text_json_history}")

        schema = StructType([
        StructField("currentTemp", DecimalType(10, 2), True), # 10 chiffres au total, dont 2 après la virgule
        StructField("analyisGPT", StringType(), True)
        ])

        df = spark.createDataFrame([(Decimal(currentTemp), analyisGPT)], schema=schema).withColumn("Datetime",current_timestamp())
      
        df.write.format("delta").mode("append").saveAsTable("Weather_Anomalies_Analysis")
    except:
        None

# MARKDOWN ********************

# ## In case of anomaly, run the pipeline with ExitValue=1 to obtain satellite images of global thermal anomalies from NASA.

# CELL ********************

GoPipeline=0
if currentTemp >=AnomalieTemp :
    GoPipeline=1
mssparkutils.notebook.exit(GoPipeline)
