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

# CELL ********************

df = spark.sql("SELECT cast(ClimateID as string) ClimateID,Name,Province FROM LK_Word_Raw.station_weather_ca where ClimateID is not null")
spark.conf.set("spark.sql.parquet.vorder.enabled", "true")
spark.conf.set("spark.microsoft.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.microsoft.delta.optimizeWrite.binSize", "1073741824")
df.write.format("delta").mode("overwrite").save("Tables/station_weather_ca_clean")
