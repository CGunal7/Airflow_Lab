from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from src.lab import (
    load_data,
    data_preprocessing,
    build_save_model,
    load_model_elbow,
    compare_with_dbscan,
)
from airflow import configuration as conf

conf.set('core', 'enable_xcom_pickling', 'True')

default_args = {
    'owner': 'your_name',
    'start_date': datetime(2023, 9, 17),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'Airflow_Lab1',
    default_args=default_args,
    description='Customer Segmentation: KMeans + DBSCAN Comparison',
    schedule_interval=None,
    catchup=False,
)

load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)

data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

build_save_model_task = PythonOperator(
    task_id='build_save_model_task',
    python_callable=build_save_model,
    op_args=[data_preprocessing_task.output, "model.sav"],
    provide_context=True,
    dag=dag,
)

load_model_task = PythonOperator(
    task_id='load_model_task',
    python_callable=load_model_elbow,
    op_args=["model.sav", build_save_model_task.output],
    dag=dag,
)

compare_models_task = PythonOperator(
    task_id='compare_models_task',
    python_callable=compare_with_dbscan,
    op_args=[data_preprocessing_task.output],
    dag=dag,
)

load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task
data_preprocessing_task >> compare_models_task

if __name__ == "__main__":
    dag.cli()



