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
# META         }
# META       ]
# META     },
# META     "environment": {
# META       "environmentId": "cf2eb93c-e6c0-408b-822b-1a52dfa7fe58",
# META       "workspaceId": "79b52fa3-7500-485b-b008-8bdaeac700d7"
# META     }
# META   }
# META }

# MARKDOWN ********************

# ## **Import de tous les elements necessaires**

# CELL ********************

import mlflow
from pyspark.sql.functions import year, month, dayofyear, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import mlflow
import mlflow.spark
import random
from datetime import datetime, timedelta
from pyspark.sql.functions import col
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.functions import monotonically_increasing_id
import mlflow
from pyspark.sql.functions import dayofweek
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, sequence, to_date, lit
from pyspark.sql.functions import year, month, dayofmonth

# MARKDOWN ********************

# ## **Set up the MLflow experiment tracking**

# CELL ********************

# Set given experiment as the active experiment. If an experiment with this name does not exist, a new experiment with this name is created.
myspace="XP_prevision_meteo"
mlflow.set_experiment(myspace)
mlflow.autolog(disable=True)  # Disable MLflow autologging

# MARKDOWN ********************

# ## **Initialisation main Dataframe**

# CELL ********************

# Générer et préparer le dataset
df = spark.sql("SELECT Datetime as date, MaxTempC, MinTempC, TotalPrecipmm, case when TotalPrecipmm>0 then 1 else 0 end as Rain  FROM LK_weather_staging.weather_canada_bc_clean")

# MARKDOWN ********************

# ## **Initialisation de la table des modeles**

# CELL ********************

targets = ["MaxTempC", "MinTempC","Rain", "TotalPrecipmm"]
models = {}

# MARKDOWN ********************

# ## 1 - Training

# MARKDOWN ********************

# ## **Detections d'anomalies with IsolationForest**

# CELL ********************

from pyspark.sql import SparkSession
from sklearn.ensemble import IsolationForest

# Initialiser Spark Session
spark = SparkSession.builder.appName("AnomalyDetection").getOrCreate()

# Charger vos données
# df = spark.read...

# Convertir Spark DataFrame en Pandas DataFrame pour utiliser Isolation Forest
# Assurez-vous que cela est faisable avec la taille de vos données
pdf = df.toPandas()

# Initialiser et ajuster le modèle Isolation Forest
clf = IsolationForest(random_state=0)
clf.fit(pdf[['TotalPrecipmm', 'MinTempC', 'MaxTempC']])

# Détecter les anomalies (les scores sont dans l'intervalle [-1, 1], où -1 indique une anomalie)
pdf['anomaly_score'] = clf.predict(pdf[['TotalPrecipmm', 'MinTempC', 'MaxTempC']])

# Convertir le Pandas DataFrame modifié en Spark DataFrame
sdf_anomalies = spark.createDataFrame(pdf)

# Afficher les résultats
sdf_anomalies.show()

# MARKDOWN ********************

# ## **Statistiques des anomalies**

# CELL ********************

# Compter le nombre d'anomalies vs observations normales
anomaly_counts = sdf_anomalies.groupBy("anomaly_score").count()

# Afficher les résultats
anomaly_counts.show()


# MARKDOWN ********************

# ## **Prepare Rain data for Random forest model**  

# CELL ********************

# Compter le nombre d'instances pour chaque classe
class_counts = df.groupBy("Rain").count().collect()
# Trouver le nombre minimal d'instances parmi les classes
min_count = min(class_counts, key=lambda x: x[1])[1]
# Sous-échantillonnage de la classe majoritaire
df_0 = df.filter(col("Rain") == 0).sample(withReplacement=False, fraction=min_count/df.filter(col("Rain") == 0).count(), seed=42)
df_1 = df.filter(col("Rain") == 1).sample(withReplacement=False, fraction=min_count/df.filter(col("Rain") == 1).count(), seed=42)
# Combiner les DataFrames sous-échantillonnés pour obtenir un ensemble équilibré
rain_df_prepared = df_0.unionAll(df_1)
balanced_df = rain_df_prepared.withColumn("date", col("date").cast("date")) \
       .withColumn("year", year("date")) \
       .withColumn("month", month("date")) \
       .withColumn("day_of_year", dayofyear("date"))
vectorAssembler = VectorAssembler(inputCols=["year", "month", "day_of_year"], outputCol="features")       
balanced_df_prepared = vectorAssembler.transform(balanced_df)


# Split des données
balanced_train_data, balanced_test_data = balanced_df_prepared.randomSplit([0.8, 0.2], seed=42)

# Classifier pour prédire la présence de pluie
lr_rain = RandomForestClassifier(labelCol="Rain", featuresCol="features")
model_rain = lr_rain.fit(balanced_train_data)
models["Rain"] =  model_rain
predictions_rain=model_rain.transform(balanced_test_data) 

# MARKDOWN ********************

# ## **Define Rain ML Model**

# CELL ********************

def Rain_Predictor():
    evaluatorBinary = BinaryClassificationEvaluator(labelCol="Rain")
    auc = evaluatorBinary.evaluate(predictions_rain, {evaluatorBinary.metricName: "areaUnderROC"})
    print(f"AUC for Rain Prediction: {auc}")
    mlflow.log_metric("Rain_AUC", auc)

# MARKDOWN ********************

# ## **Define precipitation Random Forest Model**

# CELL ********************

# Préparation des données pour TotalPrecipmm
df_precipitation = df.withColumn("date", col("date").cast("date")) \
       .withColumn("year", year("date")) \
       .withColumn("month", month("date")) \
       .withColumn("day_of_year", dayofyear("date"))

vectorAssembler = VectorAssembler(inputCols=["year", "month", "day_of_year"], outputCol="features")
precipitation_df_prepared = vectorAssembler.transform(df_precipitation.filter(col("TotalPrecipmm")>0))


# Split des données
precipitation_train_data, precipitation_test_data = precipitation_df_prepared.randomSplit([0.8, 0.2], seed=42)


def TotalPrecipmm_Predictor():
    # Prédiction de la présence de pluie
    # Filtrage des jours prédits comme pluvieux pour la régression
    predictions_rain_renamed=predictions_rain.withColumnRenamed("prediction","rain_prediction")
    pluie_predite_df = predictions_rain_renamed.filter(predictions_rain_renamed.rain_prediction == 1)

    # Régression pour prédire la quantité de précipitations
    # Assurez-vous que votre modèle de régression est entraîné sur les données avec précipitations > 0
    rf = RandomForestRegressor(featuresCol="features", labelCol="TotalPrecipmm")
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 20]) \
        .addGrid(rf.maxDepth, [5, 10]) \
        .build()

    crossval = CrossValidator(estimator=rf,
        estimatorParamMaps=paramGrid,
        evaluator=RegressionEvaluator(labelCol="label", metricName="rmse"),
        numFolds=3)

    # Entraînement du modèle de régression sur les données filtrées où TotalPrecipmm > 0
    model_precip = crossval.fit(precipitation_train_data)
    bestModel=model_precip.bestModel
    models[target] =  bestModel
    # Application du modèle de régression sur les jours prédits comme pluvieux
    predictions_precip = bestModel.transform(pluie_predite_df.filter(col("TotalPrecipmm")>0).withColumn("label", col(target)))
                    
    # Évaluation
    rmse = RegressionEvaluator(labelCol="label", metricName="rmse").evaluate(predictions_precip)
    mae = RegressionEvaluator(labelCol="label", metricName="mae").evaluate(predictions_precip)
    r2 = RegressionEvaluator(labelCol="label", metricName="r2").evaluate(predictions_precip)

    # Log dans MLflow
    mlflow.log_params({"numTrees": bestModel.getNumTrees, "maxDepth": bestModel.getOrDefault('maxDepth')})
    mlflow.log_metrics({f"{target}_RMSE": rmse, f"{target}_MAE": mae, f"{target}_R2": r2})
    mlflow.spark.log_model(bestModel, f"model_{target}")
                
    print(f"Metrics for {target}: RMSE = {rmse}, MAE = {mae}, R2 = {r2}")  

# MARKDOWN ********************

# ## Run MLFlow for prevision of precipitation with FB Prophet model

# CELL ********************

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

with mlflow.start_run():
    # Split des données
    #df_train_data, df_test_data = df.randomSplit([0.5, 0.5], seed=42)

    size = df.count() // 2 
    # Diviser le DataFrame
    df_train_data = df.limit(size)
    df_test_data = df.subtract(df_train_data).limit(size)


    # Renommer les colonnes dans le DataFrame Spark
    df_renamed = df_train_data.withColumnRenamed("Date", "ds").withColumnRenamed("TotalPrecipmm", "y")

    # Ajouter une colonne 'Rain' si elle n'est pas déjà présente, sinon renommez-la également si nécessaire
    # Si 'Rain' est déjà dans df, cette étape peut être omise ou adaptée

    # Convertir en DataFrame pandas pour Prophet
    df_pandas = df_renamed.toPandas()

    # Initialiser le modèle Prophet
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.add_regressor('Rain') # Ajouter 'Rain' comme régulateur

    # Ajuster le modèle sur les données
    model.fit(df_pandas)

    # Créer un DataFrame pour les prédictions futures. Assurez-vous que ce DataFrame contient aussi 'Rain' pour les jours futurs

    vectorAssembler = VectorAssembler(inputCols=["year", "month", "day_of_year"], outputCol="features")
    # Préparer une date manquante pour la prédiction
    missing_date_df  = df_test_data.select("date","TotalPrecipmm","Rain")
    missing_date_df = missing_date_df.withColumn("day_of_week", dayofweek("date"))
    missing_date_df=missing_date_df.withColumn("year", year("date")) \
        .withColumn("month", month("date")) \
        .withColumn("day_of_year", dayofyear("date"))
    missing_date_df_features = vectorAssembler.transform(missing_date_df)
    future=missing_date_df_features




    # Utiliser chaque modèle pour prédire la valeur à la date manquante
    for value_col, modelx in models.items():
        if value_col == "Rain":
            # Effectuer la prédiction pour la date manquante
            prediction = modelx.transform(future)
            
            # Renommer la colonne 'prediction' pour chaque valeur
            prediction = prediction.withColumnRenamed("prediction", f"{value_col}_predicted")
            
           

    # Ici, vous devez fournir des valeurs pour 'Rain' dans le DataFrame future
    future = prediction.select("date","Rain_predicted").withColumnRenamed("date", "ds").withColumnRenamed("Rain_predicted", "Rain").toPandas() # Ajustez cette valeur selon vos besoins ou votre logique de prédiction

    # Prédire les valeurs futures
    models["PrecipitationM2"] =  model
    forecast = model.predict(future)

    # Afficher les prédictions
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    y_true = df_train_data.withColumn("day_of_week", dayofweek("date")).select("day_of_week","TotalPrecipmm").toPandas()
    y_pred = prediction.select("day_of_week","TotalPrecipmm").toPandas()


    print(y_true.count())
    print(y_pred.count())
    # Calculer MAE
    mae = mean_absolute_error(y_true, y_pred)
    print(f"MAE: {mae}")

    # Calculer RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"RMSE: {rmse}")

    # Calculer R2
    r2 = r2_score(y_true, y_pred)
    print(f"R2: {r2}")

    # Loguer les métriques avec MLflow
    mlflow.log_metric("MAE_PrecipitationM2", mae)
    mlflow.log_metric("RMSE_PrecipitationM2", rmse)
    mlflow.log_metric("R2_PrecipitationM2", r2)


# MARKDOWN ********************

# ## **Define Min et Max Temperatures ML Model (Random forest with grid search)**

# CELL ********************

# Préparation des données pour MinTempC,MaxTempC
df_minmax = df.withColumn("date", col("date").cast("date")) \
       .withColumn("year", year("date")) \
       .withColumn("month", month("date")) \
       .withColumn("day_of_year", dayofyear("date"))

vectorAssembler = VectorAssembler(inputCols=["year", "month", "day_of_year"], outputCol="features")
df_prepared = vectorAssembler.transform(df_minmax)


# Split des données
train_data, test_data = df_prepared.randomSplit([0.8, 0.2], seed=42)


def MinMaxTempC_Predictor(target):
    rf = RandomForestRegressor(featuresCol="features", labelCol="label")
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 20]) \
        .addGrid(rf.maxDepth, [5, 10]) \
        .build()

    crossval = CrossValidator(estimator=rf,
                                estimatorParamMaps=paramGrid,
                                evaluator=RegressionEvaluator(labelCol="label", metricName="rmse"),
                                numFolds=3)
            
    model = crossval.fit(train_data_target)
    bestModel = model.bestModel
    models[target] =  bestModel
    predictions = bestModel.transform(test_data_target)
                
    # Évaluation
    rmse = RegressionEvaluator(labelCol="label", metricName="rmse").evaluate(predictions)
    mae = RegressionEvaluator(labelCol="label", metricName="mae").evaluate(predictions)
    r2 = RegressionEvaluator(labelCol="label", metricName="r2").evaluate(predictions)

    # Log dans MLflow
    mlflow.log_params({"numTrees": bestModel.getNumTrees, "maxDepth": bestModel.getOrDefault('maxDepth')})
    mlflow.log_metrics({f"{target}_RMSE": rmse, f"{target}_MAE": mae, f"{target}_R2": r2})
    mlflow.spark.log_model(bestModel, f"model_{target}")
            
    print(f"Metrics for {target}: RMSE = {rmse}, MAE = {mae}, R2 = {r2}")

# MARKDOWN ********************

# ## **RUN MLFLOW for rain, precipitation, mintemp and maxtemp**

# CELL ********************

for target in targets:
    
    precipitation_train_data=precipitation_train_data.withColumn("label", col(target))
    train_data_target = train_data.withColumn("label", col(target))
    test_data_target = test_data.withColumn("label", col(target))        
    
    with mlflow.start_run():
         # Entraîner un modèle pour chaque valeur cible et stocker les modèles
        
        print(f"Training model for {target}")

        if target == "Rain" :
                Rain_Predictor()
        elif target == "TotalPrecipmm" :
                TotalPrecipmm_Predictor()
        else :
                MinMaxTempC_Predictor(target)                

# MARKDOWN ********************

# ## **2 - Run models and get forecast**

# MARKDOWN ********************

# ## **Optimisation des tables**

# CELL ********************

spark.conf.set("spark.sql.parquet.vorder.enabled", "true")
spark.conf.set("spark.microsoft.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.microsoft.delta.optimizeWrite.binSize", "1073741824")

# MARKDOWN ********************

# ## **Generation table de date for forecast**

# CELL ********************

# Créer une session Spark
spark = SparkSession.builder.getOrCreate()

# Définir les dates de début et de fin
start_date = lit("2003-01-01")
end_date = lit("2033-12-31")

# Créer une DataFrame avec une colonne contenant une liste de toutes les dates
df = spark.range(1).select(explode(sequence(to_date(start_date), to_date(end_date))).alias("date"))
df = df.withColumn("year", year("date"))
df = df.withColumn("month", month("date"))
df = df.withColumn("day", dayofmonth("date"))

# Afficher la DataFrame
df.show()

df.write.mode("overwrite").format("delta").save("Tables/date_table")


# MARKDOWN ********************

# ## **Preparation du dataset de prevision pour les dates générées**

# CELL ********************

vectorAssembler = VectorAssembler(inputCols=["year", "month", "day_of_year"], outputCol="features")
# Préparer une date manquante pour la prédiction
missing_date_df  = spark.sql("SELECT  date,0 ClimateID FROM LK_weather_staging.date_table ")
missing_date_df = missing_date_df.withColumn("date", missing_date_df["date"].cast("date"))
missing_date_df = missing_date_df.withColumn("day_of_week", dayofweek("date"))
missing_date_df=missing_date_df.withColumn("year", year("date")) \
       .withColumn("month", month("date")) \
       .withColumn("day_of_year", dayofyear("date"))
missing_date_df_features = vectorAssembler.transform(missing_date_df)


# MARKDOWN ********************

# ## **Lancer la prevision des min max**

# CELL ********************

# Utiliser chaque modèle pour prédire la valeur à la date manquante
def Run_MinMaxTempC_Predictor():
    for value_col, model in models.items():
        if value_col != "TotalPrecipmm" and value_col != "PrecipitationM2":
            # Effectuer la prédiction pour la date manquante
            prediction = model.transform(missing_date_df_features)
            #print(model)
            # Renommer la colonne 'prediction' pour chaque valeur
            prediction = prediction.withColumnRenamed("prediction", f"{value_col}_predicted")
            
            # Ajouter le DataFrame de prédiction à la liste
            predictions_list.append(prediction.select("date", f"{value_col}_predicted"))  

# MARKDOWN ********************

# ## **Preparation prevision des precipitations baser sur la prevision de Rain**

# CELL ********************

# Utiliser chaque modèle pour prédire la valeur à la date manquante
for value_col, modelx in models.items():
    if value_col == "Rain":
        # Effectuer la prédiction pour la date manquante
        prediction = modelx.transform(missing_date_df_features)
            
        # Renommer la colonne 'prediction' pour chaque valeur
        prediction = prediction.withColumnRenamed("prediction", f"{value_col}_predicted")
            
           

# Ici, vous devez fournir des valeurs pour 'Rain' dans le DataFrame future
pfuture = prediction.select("date","Rain_predicted").withColumnRenamed("date", "ds").withColumnRenamed("Rain_predicted", "Rain").toPandas() # Ajustez cette valeur selon vos besoins ou votre logique de prédiction


# Utiliser chaque modèle pour prédire la valeur à la date manquante
def Run_PrecipitationV2_Predictor():
    for value_col, model in models.items():
        if value_col == "PrecipitationM2":
            # Effectuer la prédiction pour la date manquante
            prediction = model.predict(pfuture)
            #print(model)
            # Renommer la colonne 'prediction' pour chaque valeur
            spark_prediction=spark.createDataFrame(prediction)
            spark_prediction = spark_prediction.withColumnRenamed("yhat", f"{value_col}_predicted").withColumnRenamed("ds", "date")
            
            # Ajouter le DataFrame de prédiction à la liste
            predictions_list.append(spark_prediction.select("date", f"{value_col}_predicted"))  

# MARKDOWN ********************

# ## **definition prevision des precipitations**

# CELL ********************

def Run_Precipitation_Predictor():
    for value_col, model in models.items():
        if value_col == "TotalPrecipmm":    
            vectorAssembler_precip = VectorAssembler(inputCols=["year", "month", "day_of_year", "MaxTempC", "MinTempC"], outputCol="features")
            missing_date_df_features_precip =vectorAssembler_precip.transform(predicted_values_for_missing_date.withColumn("year", year("date")) \
                .withColumn("month", month("date")) \
                .withColumn("day_of_year", dayofyear("date")).withColumnRenamed("MaxTempC_predicted","MaxTempC").withColumnRenamed("MinTempC_predicted","MinTempC"))

            # Effectuer la prédiction pour la date manquante
            prediction = model.transform(missing_date_df_features_precip)
            #print(model)
            # Renommer la colonne 'prediction' pour chaque valeur
            prediction = prediction.withColumnRenamed("prediction", f"TotalPrecipmm_predicted")
                    
            # Ajouter le DataFrame de prédiction à la liste
            predictions_list.append(prediction.select("date", f"TotalPrecipmm_predicted")) 

# MARKDOWN ********************

# ## **Pipeline lancement des previsions**

# CELL ********************

# Initialisation de la liste pour stocker les DataFrames de prédictions
predictions_list = []

#Lancer les prevision pour es temperatures min et max
Run_MinMaxTempC_Predictor()     
    
# Fusionner tous les DataFrames de prédictions dans un seul DataFrame
from functools import reduce
def join_dfs(lhs, rhs):
    return lhs.join(rhs, on="date", how="inner")
predicted_values_for_missing_date = reduce(join_dfs, predictions_list)

#lancer les prevision pour les precipitations
Run_Precipitation_Predictor()

Run_PrecipitationV2_Predictor()


# Fusionner tous les DataFrames de prédictions dans un seul DataFrame
from functools import reduce
def join_dfs(lhs, rhs):
    return lhs.join(rhs, on="date", how="inner")
predicted_values_for_missing_date_precip = reduce(join_dfs, predictions_list)


#Ajustement de la prevision des Precipitation et la prevision de la pluie
predicted_values_for_missing_date_precip=predicted_values_for_missing_date_precip.withColumn("TotalPrecipmm_predictedRain",col("TotalPrecipmm_predicted")*col("Rain_predicted"))\
                                        .withColumn("PrecipitationM2_predictedRain",col("PrecipitationM2_predicted")*col("Rain_predicted"))

#Affichage des previsions
predicted_values_for_missing_date_precip.show()

#Enregistrement des previsions
predicted_values_for_missing_date_precip.write.mode("overwrite").format("delta").save("Tables/weather_canada_bc_prevision")

# MARKDOWN ********************

# ## **Statistiques des previsions**

# CELL ********************

df = spark.sql("SELECT * FROM LK_weather_staging.weather_canada_bc_prevision where year(date)<2024")
df.describe().show()

# MARKDOWN ********************

# ## **Liste des experiances lancés**

# CELL ********************

experiments = mlflow.search_experiments()
for exp in experiments:
    print(exp.name)

# MARKDOWN ********************

# ## **Identification de l'experiance actuelles**

# CELL ********************

experiment_name = myspace
exp = mlflow.get_experiment_by_name(experiment_name)
print(exp)

# MARKDOWN ********************

# ## **Historiques des runs de notre experiance**

# CELL ********************

mlflow.search_runs(exp.experiment_id)

# MARKDOWN ********************

# ## **Résultat des runs par prevision**

# CELL ********************

# Assurez-vous d'avoir configuré MLflow pour pointer vers votre serveur ou dossier de tracking

# Spécifiez l'ID de votre expérience MLflow
# Vous pouvez trouver cet ID dans l'interface utilisateur de MLflow ou via l'API
experiment_id = exp.experiment_id # Remplacez '1' par l'ID de votre expérience

# Rechercher tous les runs dans une expérience spécifique
runs = mlflow.search_runs(experiment_ids=[experiment_id]).reset_index()

# Filtrer les colonnes pour ne récupérer que celles qui nous intéressent (e.g., métriques)
metrics_columnsMax = ['metrics.MaxTempC_RMSE', 'metrics.MaxTempC_R2', 'metrics.MaxTempC_MAE','run_id','index']
runs_filteredMax = runs[metrics_columnsMax]
runs_filteredMax_clean=runs_filteredMax.dropna()
# Affichez les résultats pour vérifier
display(runs_filteredMax_clean)

# Filtrer les colonnes pour ne récupérer que celles qui nous intéressent (e.g., métriques)
metrics_columnsMin = ['metrics.MinTempC_RMSE', 'metrics.MinTempC_R2', 'metrics.MinTempC_MAE','run_id','index']
runs_filteredMin = runs[metrics_columnsMin]
runs_filteredMin_clean=runs_filteredMin.dropna()
# Affichez les résultats pour vérifier
display(runs_filteredMin_clean)

# Filtrer les colonnes pour ne récupérer que celles qui nous intéressent (e.g., métriques)
metrics_columnsTotalPrecip = ['metrics.TotalPrecipmm_RMSE', 'metrics.TotalPrecipmm_R2', 'metrics.TotalPrecipmm_MAE','run_id','index']
runs_filteredTotalPrecip = runs[metrics_columnsTotalPrecip]
runs_filteredTotalPrecip_clean=runs_filteredTotalPrecip.dropna()
# Affichez les résultats pour vérifier
display(runs_filteredTotalPrecip_clean)


# Filtrer les colonnes pour ne récupérer que celles qui nous intéressent (e.g., métriques)

runs_filtered_clean=pd.concat([runs_filteredMin_clean, runs_filteredTotalPrecip_clean, runs_filteredMax_clean]).sort_index()


# MARKDOWN ********************

# ## **Calcul du visuel qualité des MinTempC prevision**

# CELL ********************

def Show_Plot_MinTempC(runs_filtered_clean_sortedmin):
    # S'assurer que toutes les colonnes de métriques sont numériques et filtrer les NaN
    for col in ['metrics.MinTempC_RMSE', 'metrics.MinTempC_R2', 'metrics.MinTempC_MAE']:
        runs_filtered_clean_sortedmin[col] = pd.to_numeric(runs_filtered_clean_sortedmin[col], errors='coerce')

    runs_filtered_clean_sortedmin = runs_filtered_clean_sortedmin.dropna(subset=['metrics.MinTempC_RMSE', 'metrics.MinTempC_R2', 'metrics.MinTempC_MAE'])

    # Recréer les indices pour correspondre à la taille des données filtrées
    indices = range(len(runs_filtered_clean_sortedmin))

    # Tracer les métriques
    plt.figure(figsize=(15, 3))

    # RMSE
    plt.plot(indices, runs_filtered_clean_sortedmin['metrics.MinTempC_RMSE'], marker='o', linestyle='-', color='blue', label='RMSE')
    # R2
    plt.plot(indices, runs_filtered_clean_sortedmin['metrics.MinTempC_R2'], marker='s', linestyle='--', color='green', label='R2')
    # MAE
    plt.plot(indices, runs_filtered_clean_sortedmin['metrics.MinTempC_MAE'], marker='^', linestyle='-.', color='red', label='MAE')

    plt.title('Évolution des Métriques de Performance par Run TempMinC')
    plt.xlabel('Run Index')
    plt.ylabel('Valeur de Métrique')
    plt.legend()
    plt.grid(True)
    plt.xticks(indices, rotation=45) # Rotation des étiquettes de l'axe X pour améliorer la lisibilité
    plt.tight_layout()
    plt.show()

# MARKDOWN ********************

# ## **Calcul du visuel qualité des MaxTempC prevision**

# CELL ********************

def Show_Plot_MaxTempC(runs_filtered_clean_sortedmax):
    # S'assurer que toutes les colonnes de métriques sont numériques et filtrer les NaN
    for col in ['metrics.MaxTempC_RMSE', 'metrics.MaxTempC_R2', 'metrics.MaxTempC_MAE']:
        runs_filtered_clean_sortedmax[col] = pd.to_numeric(runs_filtered_clean_sortedmax[col], errors='coerce')

    runs_filtered_clean_sortedmax = runs_filtered_clean_sortedmax.dropna(subset=['metrics.MaxTempC_RMSE', 'metrics.MaxTempC_R2', 'metrics.MaxTempC_MAE'])

    # Recréer les indices pour correspondre à la taille des données filtrées
    indices = range(len(runs_filtered_clean_sortedmax))

    # Tracer les métriques
    plt.figure(figsize=(15, 3))

    # RMSE
    plt.plot(indices, runs_filtered_clean_sortedmax['metrics.MaxTempC_RMSE'], marker='o', linestyle='-', color='blue', label='RMSE')
    # R2
    plt.plot(indices, runs_filtered_clean_sortedmax['metrics.MaxTempC_R2'], marker='s', linestyle='--', color='green', label='R2')
    # MAE
    plt.plot(indices, runs_filtered_clean_sortedmax['metrics.MaxTempC_MAE'], marker='^', linestyle='-.', color='red', label='MAE')

    plt.title('Évolution des Métriques de Performance par Run TempMaxC')
    plt.xlabel('Run Index')
    plt.ylabel('Valeur de Métrique')
    plt.legend()
    plt.grid(True)
    plt.xticks(indices, rotation=45) # Rotation des étiquettes de l'axe X pour améliorer la lisibilité
    plt.tight_layout()
    plt.show()

# MARKDOWN ********************

# ## **Calcul du visuel qualité des Precipitations prevision**

# CELL ********************

def Show_Plot_TotalPrecipmm(runs_filtered_clean_sorted):
    # S'assurer que toutes les colonnes de métriques sont numériques et filtrer les NaN
    for col in ['metrics.TotalPrecipmm_RMSE', 'metrics.TotalPrecipmm_R2', 'metrics.TotalPrecipmm_MAE']:
        runs_filtered_clean_sorted[col] = pd.to_numeric(runs_filtered_clean_sorted[col], errors='coerce')

    runs_filtered_clean_sorted = runs_filtered_clean_sorted.dropna(subset=['metrics.TotalPrecipmm_RMSE', 'metrics.TotalPrecipmm_R2', 'metrics.TotalPrecipmm_MAE'])

    # Recréer les indices pour correspondre à la taille des données filtrées
    indices = range(len(runs_filtered_clean_sorted))

    # Tracer les métriques
    plt.figure(figsize=(15, 3))

    # RMSE
    plt.plot(indices, runs_filtered_clean_sorted['metrics.TotalPrecipmm_RMSE'], marker='o', linestyle='-', color='blue', label='RMSE')
    # R2
    plt.plot(indices, runs_filtered_clean_sorted['metrics.TotalPrecipmm_R2'], marker='s', linestyle='--', color='green', label='R2')
    # MAE
    plt.plot(indices, runs_filtered_clean_sorted['metrics.TotalPrecipmm_MAE'], marker='^', linestyle='-.', color='red', label='MAE')

    plt.title('Évolution des Métriques de Performance par Run TotalPrecipmm')
    plt.xlabel('Run Index')
    plt.ylabel('Valeur de Métrique')
    plt.legend()
    plt.grid(True)
    plt.xticks(indices, rotation=45) # Rotation des étiquettes de l'axe X pour améliorer la lisibilité
    plt.tight_layout()
    plt.show()

# MARKDOWN ********************

# ## **Affichage des plots**

# CELL ********************

Show_Plot_MinTempC(runs_filtered_clean)
Show_Plot_MaxTempC(runs_filtered_clean)
Show_Plot_TotalPrecipmm(runs_filtered_clean)

# MARKDOWN ********************

# ## **Preparation des données pour l'analyses Prediction VS Reel**

# CELL ********************

maxval = spark.sql("SELECT count(*) FROM LK_weather_staging.weather_canada_bc_clean")
count_result = maxval.collect()[0][0]
# Convertit la valeur en entier
maxval_entier = int(count_result)
predictions = spark.sql(f"SELECT MaxTempC_predicted AS MaxTempC, MinTempC_predicted AS MinTempC, (TotalPrecipmm_predicted * Rain_predicted) AS TotalPrecipmm FROM LK_weather_staging.weather_canada_bc_prevision LIMIT {maxval_entier}")

y_test = spark.sql(f"SELECT MaxTempC, MinTempC, TotalPrecipmm FROM LK_weather_staging.weather_canada_bc_clean LIMIT {maxval_entier}")
# Convertir les DataFrames PySpark en DataFrames pandas
predictions_df = predictions.toPandas()
y_test_df = y_test.toPandas()

# MARKDOWN ********************

# ## **Affichage du resultat Prediction VS Reel**

# CELL ********************

# Tracer les valeurs
plt.scatter(predictions_df['MaxTempC'], y_test_df['MaxTempC'], label='Température maximale')
plt.scatter(predictions_df['MinTempC'], y_test_df['MinTempC'], label='Température minimale')
plt.scatter(predictions_df['TotalPrecipmm'], y_test_df['TotalPrecipmm'], label='Précipitations totales')

# Ligne y=x pour référence
plt.plot([y_test_df.min().min(), y_test_df.max().max()], [y_test_df.min().min(), y_test_df.max().max()], 'k--', lw=4)


plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.title('Prédictions vs Valeurs Réelles')
plt.legend()
plt.show()

