"""
=====================================================================
 GOLD LAYER — Host Aggregates NO IDPs (coa_iso, year)
=====================================================================
Obiettivo:
Aggregare gold_displacement mantenendo la colonna degli IDP per 
fini di visualizzazione, ma CALCOLANDO le metriche di pressione 
(total_stock, inflows, outflows, growth_rate) ESCLUSIVAMENTE sui 
flussi transfrontalieri (Rifugiati, Richiedenti Asilo, OIP).
=====================================================================
"""

import sys
import os
import pyspark.sql.functions as F
from pyspark.sql.window import Window

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utilities import get_spark_session, initialize_delta_table

spark = get_spark_session("Gold-HostAggregates-NOidps")

# =====================================================================
# STEP 1 — LETTURA E AGGREGAZIONE CONDIZIONALE
# =====================================================================
displacement_df = spark.read.format("delta").load("s3a://lakehouse/gold/gold_displacement")

host_year_sums = displacement_df.groupBy("coa_iso", "year").agg(
    
    # 1. Spacchettamento: contiamo tutto tenendo le categorie separate
    F.sum(F.when(F.col("coo_iso") != F.col("coa_iso"), F.col("refugees")).otherwise(0)).alias("refugees_count"),
    F.sum(F.when(F.col("coo_iso") != F.col("coa_iso"), F.col("asylum_seekers")).otherwise(0)).alias("asylum_seekers_count"),
    F.sum(F.when(F.col("coo_iso") != F.col("coa_iso"), F.col("oip")).otherwise(0)).alias("oip_count"),
    
    # Colonna IDP a sé stante per la dashboard
    F.sum(F.when(F.col("coo_iso") == F.col("coa_iso"), F.col("idps")).otherwise(0)).alias("idps_count"),
    
    # 2. Metriche Totali: calcolate SOLO sulle rotte transfrontaliere
    F.sum(F.when(F.col("coo_iso") != F.col("coa_iso"), F.col("stock")).otherwise(0)).alias("total_hosted_stock"),
    F.sum(F.when(F.col("coo_iso") != F.col("coa_iso"), F.col("inflows")).otherwise(0)).alias("total_inflows"),
    F.sum(F.when(F.col("coo_iso") != F.col("coa_iso"), F.col("outflows")).otherwise(0)).alias("total_outflows"),
)

# =====================================================================
# STEP 2 — GRIGLIA DENSA NATIVA SPARK
# =====================================================================
year_bounds = displacement_df.agg(
    F.min("year").cast("integer").alias("min_year"),
    F.max("year").cast("integer").alias("max_year")
)

all_hosts_df = host_year_sums.select("coa_iso").distinct()

dense_grid = (
    all_hosts_df.crossJoin(F.broadcast(year_bounds))
    .withColumn("year_array", F.expr("sequence(min_year, max_year, 1)"))
    .withColumn("year", F.explode("year_array"))
    .drop("min_year", "max_year", "year_array")
)

# =====================================================================
# STEP 3 — JOIN E RIEMPIMENTO A ZERO
# =====================================================================
host_panel = (
    dense_grid
    .join(host_year_sums, ["coa_iso", "year"], "left")
    .fillna(0, subset=[
        "refugees_count", "asylum_seekers_count", "oip_count", "idps_count",
        "total_hosted_stock", "total_inflows", "total_outflows"
    ])
)

# =====================================================================
# STEP 4 — LAG E GROWTH RATE (Calcolato senza gli IDP)
# =====================================================================
window_host = Window.partitionBy("coa_iso").orderBy("year")

host_panel = host_panel.withColumn(
    "hosted_stock_lag1",
    F.lag("total_hosted_stock", 1).over(window_host)
)

host_panel = host_panel.withColumn(
    "growth_rate",
    F.when(F.col("hosted_stock_lag1").isNull(), F.lit(None).cast("double"))
     .when((F.col("hosted_stock_lag1") == 0) & (F.col("total_hosted_stock") == 0), F.lit(0.0))
     .when((F.col("hosted_stock_lag1") == 0) & (F.col("total_hosted_stock") > 0), F.lit(None).cast("double"))
     .otherwise(
        (F.col("total_hosted_stock") - F.col("hosted_stock_lag1")) / F.col("hosted_stock_lag1")
     )
)

# =====================================================================
# STEP 5 — SELEZIONE E SCRITTURA SU DELTA LAKE
# =====================================================================
gold_host_aggregates_noidps = host_panel.select(
    "coa_iso",
    "year",
    "refugees_count",
    "asylum_seekers_count",
    "oip_count",
    "idps_count", 
    "total_hosted_stock",
    "hosted_stock_lag1",
    "total_inflows",
    "total_outflows",
    "growth_rate",
)

initialize_delta_table(
    spark=spark,
    db_name="gold",
    table_name="gold_host_aggregates_noidps"
)

print("Scrittura della tabella gold_host_aggregates_noidps su Delta Lake...")
(
    gold_host_aggregates_noidps.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save("s3a://lakehouse/gold/gold_host_aggregates_noidps")
)

print("Tabella gold_host_aggregates_noidps generata con successo.")
spark.stop()