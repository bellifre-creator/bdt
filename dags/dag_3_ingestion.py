from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'retries': 0,
}

with DAG(
    dag_id='3_silver_to_gold',
    default_args=default_args,
    #schedule=timedelta(minutes=5),
    start_date=datetime(2026, 6, 20),
    catchup=False,
    max_active_runs=1,
) as dag:

    # # 1. Gold task for conflict features
    # silver_to_gold_conflict = SparkSubmitOperator(
    #     task_id='silver_to_gold_conflict_task',
    #     application='/opt/spark/jobs/gold/gold_conflict_features.py',
    #     conn_id='spark_default',
    #     packages='io.delta:delta-spark_2.12:3.2.0',
    # )

    # silver_to_gold_poverty = SparkSubmitOperator(
    #     task_id='silver_to_gold_poverty_task',
    #     application='/opt/spark/jobs/gold/gold_poverty_features.py',
    #     conn_id='spark_default',
    #     packages='io.delta:delta-spark_2.12:3.2.0',
    # )

    # silver_to_gold_food_security = SparkSubmitOperator(
    #     task_id='silver_to_gold_food_security_task',
    #     application='/opt/spark/jobs/gold/gold_food_security_features.py',
    #     conn_id='spark_default',
    #     packages='io.delta:delta-spark_2.12:3.2.0',
    # )

    # silver_to_gold_host_press_matrix = SparkSubmitOperator(
    #     task_id='silver_to_gold_host_press_matrix',
    #     application='/opt/spark/jobs/gold/silver-to-gold_master_matrix.py',
    #     conn_id='spark_default',
    #     packages='io.delta:delta-spark_2.12:3.2.0',
    # )

    silver_to_gold_ML = SparkSubmitOperator(
        task_id='silver_to_gold_ML_task',
        application='/opt/spark/jobs/gold/silver_to_gold_host_pressure.py',
        conn_id='spark_default',
        packages='io.delta:delta-spark_2.12:3.2.0',
    )

    silver_to_gold_ML

    # silver_to_gold_conflict>>silver_to_gold_poverty>>silver_to_gold_food_security>>silver_to_gold_host_press_matrix

    # # Task D: Aggregate the data (Silver -> Gold)
    # silver_to_gold_t = SparkSubmitOperator(
    #     task_id='silver_to_gold_t',
    #     application='/opt/spark/jobs/gold/silver-to-gold-test2.py',
    #     conn_id='spark_default',
    #     packages='io.delta:delta-spark_2.12:3.2.0',
    # )



    # silver_to_gold_t>>silver_to_gold_ad