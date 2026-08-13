# import sys
# import os
# import pyspark.sql.functions as F
# from pyspark.ml.feature import VectorAssembler
# from pyspark.ml.regression import RandomForestRegressor
# from pyspark.ml.evaluation import RegressionEvaluator

# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(parent_dir)

# from utilities import get_spark_session, initialize_delta_table

# spark = get_spark_session("ML-HostPressure-Model")

# print("1. Lettura della Master Matrix dal layer Gold...")
# df = spark.read.format("delta").load("s3a://lakehouse/gold/gold_master_matrix")

# # 2. Gestione dei dati mancanti e preparazione
# print("2. Trattamento dei Null (Out-of-Distribution Imputation)...")

# # Le colonne continue che potrebbero avere nulli non per assenza di evento, ma per assenza di misurazione
# cols_with_missing_data = [
#     "push_mpi", "push_mpm", "push_ext_pov", "push_food_crisis_pop",
#     "pull_gdp_per_capita", "pull_host_capacity"
# ]

# # Riempiamo i nulli con -1 per permettere alla Random Forest di isolarli
# df_clean = df.fillna(-1, subset=cols_with_missing_data)

# # Eliminiamo le righe dove il target (Y) o la diaspora storica (variabile più importante) sono nulli
# # Non possiamo addestrare o prevedere su queste righe.
# df_clean = df_clean.dropna(subset=["target_host_pressure_stock", "pull_historical_diaspora"])

# # 3. Selezione delle Features (X) e Target (Y)
# feature_cols = [
#     "pull_historical_diaspora",
#     "flow_returned_refugees",
#     "flow_resettlement",
#     "push_idps_stock",
#     "push_violent_events",
#     "push_total_fatalities",
#     "push_civilian_targeting",
#     "push_mpi",
#     "push_mpm",
#     "push_ext_pov",
#     "push_food_crisis_pop",
#     "pull_gdp_per_capita",
#     "pull_host_capacity",
#     "pull_funding_received_usd"
# ]

# # Assemblatore vettoriale: trasforma le singole colonne in un unico vettore "features" richiesto da PySpark ML
# assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
# model_data = assembler.transform(df_clean)

# # 4. Split Temporale (Time-Series Split)
# print("3. Suddivisione in Train (<= 2023) e Test (>= 2024)...")
# train_data = model_data.filter(F.col("year") <= 2023)
# test_data = model_data.filter(F.col("year") >= 2024)

# # 5. Addestramento del Modello (Random Forest Regressor)
# print("4. Addestramento della Random Forest...")
# rf = RandomForestRegressor(
#     featuresCol="features", 
#     labelCol="target_host_pressure_stock",
#     numTrees=100,
#     maxDepth=10,
#     seed=42
# )
# rf_model = rf.fit(train_data)

# # 6. Previsione e Valutazione
# print("5. Valutazione sul Test Set...")
# predictions = rf_model.transform(test_data)

# evaluator_rmse = RegressionEvaluator(
#     labelCol="target_host_pressure_stock", predictionCol="prediction", metricName="rmse"
# )
# evaluator_r2 = RegressionEvaluator(
#     labelCol="target_host_pressure_stock", predictionCol="prediction", metricName="r2"
# )
# evaluator_mae = RegressionEvaluator(
#     labelCol="target_host_pressure_stock", predictionCol="prediction", metricName="mae"
# )

# rmse = evaluator_rmse.evaluate(predictions)
# r2 = evaluator_r2.evaluate(predictions)
# mae = evaluator_mae.evaluate(predictions)

# print("\n--- RISULTATI DEL MODELLO ---")
# print(f"R-Squared (R2): {r2:.4f} (Vicinanza a 1 indica alta precisione)")
# print(f"RMSE: {rmse:,.2f} persone (Errore quadratico medio)")
# print(f"MAE: {mae:,.2f} persone (Errore medio assoluto)")

# # 7. Estrazione dell'importanza delle Features
# print("\n--- IMPORTANZA DELLE VARIABILI (FEATURE IMPORTANCE) ---")
# importances = rf_model.featureImportances.toArray()
# feat_importance = list(zip(feature_cols, importances))
# feat_importance.sort(key=lambda x: x[1], reverse=True)

# for feat, imp in feat_importance:
#     print(f"- {feat}: {imp:.4f}")

# # ... [codice precedente dello script con evaluator e print] ...

# print("\n6. Generazione e salvataggio delle predizioni per Superset...")

# # Applichiamo il modello addestrato su tutto il dataset (storico + test)
# # Questo ci permette di visualizzare su Superset come il modello si adatta 
# # al passato e come prevede il 2024-2025.
# all_predictions = rf_model.transform(model_data)

# # Selezioniamo solo le chiavi dimensionali e le metriche che ci interessano per la dashboard
# results_df = all_predictions.select(
#     "year",
#     "coo_iso",
#     "coa_iso",
#     F.col("target_host_pressure_stock").alias("actual_host_pressure"),
#     F.col("prediction").alias("predicted_host_pressure")
# )

# # Inizializziamo la nuova tabella Gold per le predizioni
# initialize_delta_table(
#     spark=spark,
#     db_name="gold",
#     table_name="predictions_host_pressure"
# )

# # Scriviamo il risultato su MinIO
# (
#     results_df.write
#     .format("delta")
#     .mode("overwrite")
#     .option("overwriteSchema", "true")
#     .save("s3a://lakehouse/gold/predictions_host_pressure")
# )

# print("Predizioni salvate con successo in gold.predictions_host_pressure!")
# spark.stop()


import sys
import os
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utilities import get_spark_session, initialize_delta_table

spark = get_spark_session("ML-HostPressure-Model-Advanced")

print("1. Lettura della Master Matrix dal layer Gold...")
df = spark.read.format("delta").load("s3a://lakehouse/gold/gold_master_matrix")

print("2. Trattamento dei Null (Out-of-Distribution Imputation)...")
cols_with_missing_data = [
    "push_mpi", "push_mpm", "push_ext_pov", "push_food_crisis_pop",
    "pull_gdp_per_capita", "pull_host_capacity"
]
df_clean = df.fillna(-1, subset=cols_with_missing_data)
df_clean = df_clean.dropna(subset=["target_host_pressure_stock", "pull_historical_diaspora"])

# ---------------- NUOVO TRUCCO: TRASFORMAZIONE LOGARITMICA ----------------
# Calcoliamo il log(1 + x) per normalizzare la "coda lunga" dei rifugiati
df_clean = df_clean.withColumn("log_target", F.log1p(F.col("target_host_pressure_stock")))
# ---------------------------------------------------------------------------

feature_cols = [
    "pull_historical_diaspora", "flow_returned_refugees", "flow_resettlement",
    "push_idps_stock", "push_violent_events", "push_total_fatalities",
    "push_civilian_targeting", "push_mpi", "push_mpm", "push_ext_pov",
    "push_food_crisis_pop", "pull_gdp_per_capita", "pull_host_capacity",
    "pull_funding_received_usd"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
model_data = assembler.transform(df_clean)

print("3. Suddivisione in Train (<= 2023) e Test (2024-2025)...")
train_data = model_data.filter(F.col("year") <= 2023)
test_data = model_data.filter(F.col("year") >= 2024)

# ---------------- CAMBIO ALGORITMO: GBT Regressor ----------------
print("4. Addestramento del Gradient Boosted Trees (GBT)...")
gbt = GBTRegressor(
    featuresCol="features", 
    labelCol="log_target", # Alleniamo sul logaritmo!
    maxIter=50,            # Numero di alberi iterativi
    maxDepth=5,
    seed=42
)
gbt_model = gbt.fit(train_data)
# -----------------------------------------------------------------

print("5. Valutazione sul Test Set...")
predictions_log = gbt_model.transform(test_data)

# Riportiamo le predizioni dal logaritmo al numero reale (esponenziale - 1)
predictions = predictions_log.withColumn("prediction", F.expm1(F.col("prediction")))

# Valutazione
evaluator_r2 = RegressionEvaluator(labelCol="target_host_pressure_stock", predictionCol="prediction", metricName="r2")
evaluator_mae = RegressionEvaluator(labelCol="target_host_pressure_stock", predictionCol="prediction", metricName="mae")

r2 = evaluator_r2.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)

print("\n--- NUOVI RISULTATI DEL MODELLO ---")
print(f"R-Squared (R2): {r2:.4f}")
print(f"MAE: {mae:,.2f} persone (Errore medio assoluto)")

print("\n6. FORECASTING DEL FUTURO: Previsione per il 2026...")

# Per prevedere il 2026, prendiamo l'ultima situazione nota (il 2025)
# Cambiamo la colonna dell'anno a "2026" e chiediamo al modello di stimare la pressione
future_2026_data = model_data.filter(F.col("year") == 2025).withColumn("year", F.lit(2026))

# Applichiamo il modello ai dati storici + ai dati fittizi del 2026
storico_log = gbt_model.transform(model_data)
futuro_log = gbt_model.transform(future_2026_data)

# Uniamo storico e futuro
all_predictions_log = storico_log.unionByName(futuro_log)

# Riportiamo tutto in scala reale
all_predictions = all_predictions_log.withColumn("predicted_host_pressure", F.expm1(F.col("prediction")))

results_df = all_predictions.select(
    "year",
    "coo_iso",
    "coa_iso",
    F.col("target_host_pressure_stock").alias("actual_host_pressure"),
    F.round("predicted_host_pressure").alias("predicted_host_pressure") # Arrotondiamo per non avere persone con la virgola
)

initialize_delta_table(spark=spark, db_name="gold", table_name="predictions_host_pressure")

(
    results_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save("s3a://lakehouse/gold/predictions_host_pressure")
)

print("\nPredizioni (incluso il 2026) salvate con successo in gold.predictions_host_pressure!")
spark.stop()