import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, year, to_timestamp

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utilities import get_spark_session

spark = get_spark_session("SilverToGold_MasterJoin")

# 1. Lettura delle tabelle Silver (Batch Read, NON Stream)
needs_df = spark.read.format("delta").load("s3a://lakehouse/silver/humanitarian_needs")
idps_df = spark.read.format("delta").load("s3a://lakehouse/silver/idps")
pop_df = spark.read.format("delta").load("s3a://lakehouse/silver/population")
sol_df = spark.read.format("delta").load("s3a://lakehouse/silver/solutions")

# 2. Standardizzazione delle chiavi di JOIN (country_code e year)
# Per HDX: Estraiamo l'anno dal timestamp
needs_df = needs_df.withColumn("year", year(to_timestamp(col("reference_period_start")))) \
                   .withColumnRenamed("location_code", "country_code")

idps_df = idps_df.withColumn("year", year(to_timestamp(col("reference_period_start")))) \
                 .withColumnRenamed("location_code", "country_code")

# Per UNHCR: Usiamo il Country of Origin (coo_iso) come paese principale interessato dalla crisi
pop_df = pop_df.withColumnRenamed("coo_iso", "country_code")
sol_df = sol_df.withColumnRenamed("coo_iso", "country_code")

# 3. Aggregazione per Paese e Anno
# Se non raggruppassimo, una JOIN tra i vari settori di HDX moltiplicherebbe le righe.
needs_agg = needs_df.groupBy("country_code", "year") \
                    .agg(F.sum("population").alias("total_needs_population"))

idps_agg = idps_df.groupBy("country_code", "year") \
                  .agg(F.sum("population").alias("total_idps"))

pop_agg = pop_df.groupBy("country_code", "year") \
                .agg(F.sum("refugees").alias("total_refugees"),
                     F.sum("asylum_seekers").alias("total_asylum_seekers")) ##########

sol_agg = sol_df.groupBy("country_code", "year") \
                .agg(F.sum("returned_refugees").alias("total_returns"),
                     F.sum("resettlement").alias("total_resettlement")) ############

# 4. FULL OUTER JOIN delle 4 tabelle
# Usiamo full_outer per non perdere dati se un paese è presente in un'API ma non nell'altra
gold_df = pop_agg.join(needs_agg, ["country_code", "year"], "full_outer") \
                 .join(idps_agg, ["country_code", "year"], "full_outer") \
                 .join(sol_agg, ["country_code", "year"], "full_outer")

# Riempiamo i valori nulli (generati dalla JOIN) con 0
gold_df = gold_df.fillna(0)

# 5. Scrittura della tabella Master Gold
spark.sql("CREATE DATABASE IF NOT EXISTS gold")
spark.sql("""
    CREATE TABLE IF NOT EXISTS gold.displacement_master
    USING delta
    LOCATION 's3a://lakehouse/gold/displacement_master'
""")

# Scriviamo in sovrascrittura: ogni volta che gira, rigenera la tabella finale aggiornata
gold_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .save("s3a://lakehouse/gold/displacement_master")