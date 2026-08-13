from pyspark.sql import functions as F
from pyspark.sql.window import Window
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utilities import get_spark_session

spark = get_spark_session("Gold-HostRegionPressure")

# 1. Lettura tabelle Silver (UNHCR Population e Solutions)
pop_df = spark.read.format("delta").load("s3a://lakehouse/silver/population")
sol_df = spark.read.format("delta").load("s3a://lakehouse/silver/solutions")

# 2. Funzione di supporto per la pulizia dei campi numerici
def clean_num(col_name):
    return F.coalesce(F.col(col_name).cast("long"), F.lit(0))

# 3. Aggregazione Stock per Paese d'Asilo (coa_iso) e Anno
host_stock = (
    pop_df
    .groupBy(F.col("coa_iso").alias("country_code"), F.col("year"))
    .agg(
        F.sum(clean_num("refugees")).alias("refugees_count"),
        F.sum(clean_num("asylum_seekers")).alias("asylum_seekers_count"),
        F.sum(clean_num("oip")).alias("oip_count"),
        F.sum(clean_num("idps")).alias("idps_count")
    )
    .withColumn(
        "total_forcibly_displaced_stock",
        F.col("refugees_count") + F.col("asylum_seekers_count") + 
        F.col("oip_count") + F.col("idps_count")
    )
)

# 4. Aggregazione Soluzioni (Outflows) per Paese d'Asilo e Anno
host_solutions = (
    sol_df
    .groupBy(F.col("coa_iso").alias("country_code"), F.col("year"))
    .agg(
        (F.sum(clean_num("returned_refugees")) + 
         F.sum(clean_num("resettlement")) + 
         F.sum(clean_num("naturalisation"))).alias("total_outflows")
    )
)

# 5. Join e calcolo metriche temporali
window_spec = Window.partitionBy("country_code").orderBy("year")

gold_host_pressure = (
    host_stock
    .join(host_solutions, ["country_code", "year"], "left")
    .fillna(0, subset=["total_outflows"])
    .withColumn("previous_year_stock", F.lag("total_forcibly_displaced_stock", 1).over(window_spec))
    .withColumn(
        "short_term_inflow_delta",
        F.when(F.col("previous_year_stock").isNotNull(),
               F.col("total_forcibly_displaced_stock") - F.col("previous_year_stock"))
         .otherwise(0)
    )
    .withColumn(
        "short_term_growth_pct",
        F.when((F.col("previous_year_stock").isNotNull()) & (F.col("previous_year_stock") > 0),
               ((F.col("total_forcibly_displaced_stock") - F.col("previous_year_stock")) / F.col("previous_year_stock")) * 100)
         .otherwise(0.0)
    )
    .withColumn("net_short_term_pressure", F.col("short_term_inflow_delta") - F.col("total_outflows"))
)

# 6. Registrazione automatica nel Metastore di Spark/Trino
gold_path = "s3a://lakehouse/gold/fact_host_region_pressure"

spark.sql("CREATE DATABASE IF NOT EXISTS gold")
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS gold.fact_host_region_pressure
    USING DELTA
    LOCATION '{gold_path}'
""")

# 7. Scrittura dei dati su MinIO
(
    gold_host_pressure.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(gold_path)
)

print("Tabella gold.fact_host_region_pressure creata e registrata nel metastore con successo.")