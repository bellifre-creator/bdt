import sys
import os
import pyspark.sql.functions as F
from pyspark.sql.window import Window

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utilities import get_spark_session, initialize_delta_table

spark = get_spark_session("SilverToGold-MasterMatrix")

print("1. Lettura delle tabelle dal Datalake...")
# Base (Stock e Flow)
pop_df = spark.read.format("delta").load("s3a://lakehouse/silver/population")
sol_df = spark.read.format("delta").load("s3a://lakehouse/silver/solutions")

# Push Factors (Gold)
conflict_df = spark.read.format("delta").load("s3a://lakehouse/gold/gold_conflict_features")
poverty_df = spark.read.format("delta").load("s3a://lakehouse/gold/gold_poverty_features")
food_df = spark.read.format("delta").load("s3a://lakehouse/gold/gold_food_security_features")

# Pull Factors
gdp_df = spark.read.format("delta").load("s3a://lakehouse/silver/worldbank_gdp")
wb_pop_df = spark.read.format("delta").load("s3a://lakehouse/silver/worldbank_population")
funding_df = spark.read.format("delta").load("s3a://lakehouse/gold/gold_funding_features")
location_df = spark.read.format("delta").load("s3a://lakehouse/silver/location")

print("2. Creazione della matrice base (Rotte Transfrontaliere: COO -> COA)...")
# # Filtriamo per avere solo rotte internazionali (escludiamo COO == COA)
# # Calcoliamo lo STOCK totale (Variabile Target Y)
# routes_df = pop_df.filter(F.col("coo_iso") != F.col("coa_iso")) \
#     .groupBy("year", "coo_iso", "coa_iso") \
#     .agg(
#         F.sum("refugees").alias("refugees"),
#         F.sum("asylum_seekers").alias("asylum_seekers"),
#         F.sum("oip").alias("oip")
#     ) \
#     .fillna(0, subset=["refugees", "asylum_seekers", "oip"]) \
#     .withColumn("target_host_pressure_stock", F.col("refugees") + F.col("asylum_seekers") + F.col("oip"))

# 2. Creazione della matrice base (Rotte Transfrontaliere: COO -> COA)
# Filtriamo i flussi interni e aggreghiamo per Anno, Origine e Destinazione
routes_df = (
    pop_df
    .filter(F.col("coo_iso") != F.col("coa_iso"))
    .groupBy("year", "coo_iso", "coa_iso")
    .agg(
        # Usiamo coalesce direttamente nella somma per evitare null fin dall'inizio
        F.coalesce(F.sum("refugees"), F.lit(0)).alias("refugees"),
        F.coalesce(F.sum("asylum_seekers"), F.lit(0)).alias("asylum_seekers"),
        F.coalesce(F.sum("oip"), F.lit(0)).alias("oip")
    )
    # Calcoliamo subito il Target Stock, sapendo che non ci saranno nulli
    .withColumn(
        "target_host_pressure_stock", 
        F.col("refugees") + F.col("asylum_seekers") + F.col("oip")
    )
    # Opzionale ma consigliato: drop delle colonne parziali se non ti servono per il ML
    # .drop("refugees", "asylum_seekers", "oip") 
)

# Calcolo del PULL FACTOR STORICO (Inerzia / Diaspora)
# Prendiamo la Host Pressure dell'anno precedente per la stessa rotta
window_route = Window.partitionBy("coo_iso", "coa_iso").orderBy("year")
routes_df = routes_df.withColumn(
    "pull_historical_diaspora", 
    F.lag("target_host_pressure_stock", 1).over(window_route)
)

print("3. Estrazione dei FLOWS (Rientri come valvola di sfogo)...")
# Aggreghiamo i Solutions (Flow) per la stessa tratta
flows_df = sol_df.filter(F.col("coo_iso") != F.col("coa_iso")) \
    .groupBy("year", "coo_iso", "coa_iso") \
    .agg(
        F.sum("returned_refugees").alias("flow_returned_refugees"),
        F.sum("resettlement").alias("flow_resettlement"),
        F.sum("naturalisation").alias("flow_naturalisation")
    )

print("4. Preparazione PUSH FACTORS (Origine - COO)...")
# Sfollati interni (IDPs) presi da UNHCR Population dove COO == COA
idps_coo = pop_df.filter(F.col("coo_iso") == F.col("coa_iso")) \
    .groupBy("year", "coo_iso") \
    .agg(F.sum("idps").alias("push_idps_stock"))

# Rinominiamo le colonne per evitare conflitti nella Master Join
conflict_coo = conflict_df.select(
    F.col("year"), F.col("location_code").alias("coo_iso"),
    F.col("violent_events").alias("push_violent_events"),
    F.col("total_fatalities").alias("push_total_fatalities"),
    F.col("civilian_targeting_events").alias("push_civilian_targeting")
)

poverty_coo = poverty_df.select(
    F.col("year"), F.col("location_code").alias("coo_iso"),
    F.col("mpi").alias("push_mpi"),
    F.col("mpm").alias("push_mpm"),
    F.col("ext_pov").alias("push_ext_pov")
)

food_coo = food_df.select(
    F.col("year"), F.col("location_code").alias("coo_iso"),
    F.col("peak_population_phase3plus").alias("push_food_crisis_pop")
)

print("5. Preparazione PULL FACTORS (Destinazione - COA)...")
gdp_coa = gdp_df.select(
    F.col("year"), F.col("location_code").alias("coa_iso"),
    F.col("gdp_per_capita").alias("pull_gdp_per_capita")
)

wb_pop_coa = wb_pop_df.select(
    F.col("year"), F.col("location_code").alias("coa_iso"),
    F.col("total_population").alias("pull_host_capacity")
)

# funding_coa = funding_df.groupBy("year", F.col("location_code").alias("coa_iso")).agg(
#     F.sum("funding_usd").alias("pull_funding_received_usd")
# )

# Rinominiamo le colonne per adattarle al ruolo di "Pull Factors" per il modello
funding_coa = funding_df.select(
    F.col("location_code").alias("coa_iso"), # coa_iso = Country of Asylum (Paese di Destinazione)
    F.col("year").alias("year"),
    F.col("funding_received_usd").alias("pull_funding_received_usd"),
    F.col("requirements_usd").alias("pull_requirements_usd"),
    F.col("funding_coverage_pct").alias("pull_funding_coverage_pct")
)

# Flag Statiche della location (non dipendono dall'anno, prendiamo l'ultima versione disponibile)
location_coa = location_df.select(
    F.col("code").alias("coa_iso"),
    F.col("has_hrp").alias("pull_has_hrp"),
    F.col("in_gho").alias("pull_in_gho")
).dropDuplicates(["coa_iso"])

print("6. Esecuzione della MASTER JOIN...")
# Uniamo la matrice base con tutte le features
master_matrix = routes_df \
    .join(flows_df, ["year", "coo_iso", "coa_iso"], "left") \
    .join(idps_coo, ["year", "coo_iso"], "left") \
    .join(conflict_coo, ["year", "coo_iso"], "left") \
    .join(poverty_coo, ["year", "coo_iso"], "left") \
    .join(food_coo, ["year", "coo_iso"], "left") \
    .join(gdp_coa, ["year", "coa_iso"], "left") \
    .join(wb_pop_coa, ["year", "coa_iso"], "left") \
    .join(funding_coa, ["year", "coa_iso"], "left") \
    .join(location_coa, ["coa_iso"], "left")

# Pulizia finale dei nulli generati dalle JOIN (solo per i dati quantitativi continui)
fill_dict = {
    "flow_returned_refugees": 0,
    "flow_resettlement": 0,
    "push_idps_stock": 0,
    "push_violent_events": 0,
    "push_total_fatalities": 0,
    "push_civilian_targeting": 0,
    "flow_naturalisation": 0#,
    #"pull_funding_received_usd": 0.0
}
master_matrix = master_matrix.fillna(fill_dict)

initialize_delta_table(
    spark=spark,
    db_name="gold",
    table_name="gold_master_matrix"
)

print("7. Scrittura della Matrice su Delta Lake...")
(
    master_matrix.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save("s3a://lakehouse/gold/gold_master_matrix")
)

print("Matrice Origine-Destinazione generata con successo!")
spark.stop()