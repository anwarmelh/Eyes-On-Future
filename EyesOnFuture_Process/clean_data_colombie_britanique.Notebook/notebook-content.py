# Synapse Analytics notebook source

# METADATA ********************

# META {
# META   "synapse": {
# META     "lakehouse": {
# META       "default_lakehouse": "86a42fe5-0575-4f6f-afc7-37a57a6f61ff",
# META       "default_lakehouse_name": "LK_weather_staging",
# META       "default_lakehouse_workspace_id": "0e9e1e73-833b-4f95-ac48-74db3d1ccde0",
# META       "known_lakehouses": [
# META         {
# META           "id": "86a42fe5-0575-4f6f-afc7-37a57a6f61ff"
# META         },
# META         {
# META           "id": "9732e201-a3a7-49e5-bc5f-2fe15e8c54ab"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# # Clean the data before saving it in the analysis warehouse.

# MARKDOWN ********************

# ## Import library

# CELL ********************

from pyspark.sql.functions import to_date
from pyspark.sql.functions import col, count, when, isnan, trim

# MARKDOWN ********************

# ## Select data

# CELL ********************

df = spark.sql("SELECT * FROM LK_weather_raw.weather_canada_bc")
display(df)

# MARKDOWN ********************

# ## Clean staces and Null

# CELL ********************

last_df=df.select([
    count(
        when(
            (col(c).isNull()) | 
            (isnan(c)) | 
            (trim(col(c)) == ""), # Ajout pour détecter les chaînes vides ou les espaces après avoir retiré les espaces de début et de fin
            c
        )
    ).alias(c) 
    for c in df.columns
])
display(last_df)

# MARKDOWN ********************

# ## Remove duplicate

# CELL ********************

df_selected=df.select("ClimateID","DateTime","Latitudey","MaxTempC","MinTempC","MeanTempC","TotalPrecipmm","Longitudex")
# Suppression des doublons
df_selected = df_selected.dropDuplicates()
display(df_selected.count())

# MARKDOWN ********************

# ## Remove empty

# CELL ********************

filtred_df=df_selected.filter(df_selected["MaxTempC"]!="").filter(df_selected["MinTempC"]!="")
display(filtred_df.count())

# MARKDOWN ********************

# ## Use 0.0 if no precipitation data

# CELL ********************

# Définir la valeur par défaut pour les chaînes vides
defaultValue = "0.0"

# Appliquer le remplacement pour chaque colonne de type chaîne de caractères
df_cleaned = filtred_df.withColumn("TotalPrecipmm", when(trim(col("TotalPrecipmm")) == "", defaultValue).otherwise(col("TotalPrecipmm")))
  
display(df_cleaned)


# MARKDOWN ********************

# ## Format date

# CELL ********************

df = df_cleaned.withColumn("Datetime", to_date("Datetime", "yyyy-MM-dd")) 
display(df)


# MARKDOWN ********************

# ## Optimize delta tables

# CELL ********************

spark.conf.set("spark.sql.parquet.vorder.enabled", "true")
spark.conf.set("spark.microsoft.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.microsoft.delta.optimizeWrite.binSize", "1073741824")


# MARKDOWN ********************

# ## Prepare data model

# CELL ********************

from pyspark.sql.types import StructType,StructField,IntegerType,DecimalType,DateType
 

df=df.withColumn("ClimateID",col("ClimateID").cast(IntegerType()))\
    .withColumn("Latitudey",col("Latitudey").cast(DecimalType(18,2)))\
    .withColumn("MaxTempC",col("MaxTempC").cast(DecimalType(18,2)))\
    .withColumn("MinTempC",col("MinTempC").cast(DecimalType(18,2)))\
    .withColumn("MeanTempC",col("MeanTempC").cast(DecimalType(18,2)))\
    .withColumn("TotalPrecipmm",col("TotalPrecipmm").cast(DecimalType(18,2)))\
    .withColumn("Longitudex",col("Longitudex").cast(DecimalType(18,2)))

display(df)

# CELL ********************

df.write.format("delta").mode("overwrite").save("Tables/weather_canada_bc_clean")/
(spark.read.table("LK_Word_Staging.weather_canada_bc_clean")/
.withColumn("DatetimeStrg", col("Datetime").cast("string"))/
  .write/
  .mode("overwrite")/
  .option("overwriteSchema", "true")/
  .saveAsTable("LK_Word_Staging.weather_canada_bc_clean")

# CELL ********************

%pip install openai==1.12.0

# CELL ********************

import requests
import openai
from openai import AzureOpenAI
api_key= '41e0a22e-2816-4a92-b624-5a833dd0a6b6'
url="https://polite-ground-030dc3103.4.azurestaticapps.net/api/v1"

client = AzureOpenAI(
    azure_endpoint=url,
    api_key=api_key,
    api_version="2023-09-01-preview",
)

MESSAGES = [
    {"role": "system", "content": """You are expert data scientist"""}
    ]




def callchatgpt(checkrow):
    MESSAGES.append({"role": "user", "content": "Analyses les données suivante et propose moi le meilleur modele ML  appliquer pour chacune des trois mesures TOtalPrecipmm, MinTempC et MaxTempC. Je veux un reponse claire et preise et courte en moins de 4000 tokens :"+checkrow})
    completion = client.chat.completions.create(model="gpt-4", messages=MESSAGES,temperature=0.9)
    print(completion.choices[0].message.content)

# CELL ********************

df = spark.sql("SELECT * FROM LK_Word_Staging.weather_canada_bc_clean")
display(df)

# CELL ********************


# Analyse exploratoire des données
print("Analyse exploratoire des données:")
descriptiondf=df.describe(["TotalPrecipmm"])
display(descriptiondf)


# CELL ********************

data_list=df.describe().collect()
text_output=""
for row in data_list:
    row_dict=row.asDict()
    text_output+=str(row_dict)+"\n"
row_data=f"{text_output} Count TotalPrecipmm = 0 : {df.filter(df['TotalPrecipmm'] == 0).count()}"

callchatgpt(row_data)
