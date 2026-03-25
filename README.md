# Airflow Lab 1 — Customer Segmentation Pipeline

## Overview
This lab demonstrates how to build and orchestrate a Machine Learning pipeline using **Apache Airflow** running inside **Docker**. The pipeline performs customer segmentation using clustering algorithms and compares their performance.

## Modifications from Original Lab
| | Original Lab | This Version |
|---|---|---|
| Dataset | Generic file.csv | Customer retail dataset (age, income, spending score, purchase frequency) |
| Models | K-Means only | **K-Means + DBSCAN comparison** |
| DAG Tasks | 4 tasks | **5 tasks** (added model comparison task) |
| Output | Optimal cluster count | **Best model recommendation + silhouette scores** |

## Pipeline DAG Flow
```
load_data_task
      ↓
data_preprocessing_task
      ↓                ↘
build_save_model_task   compare_models_task
      ↓
load_model_task
```

## Task Descriptions
1. **load_data_task** — Loads customer CSV data (30 customers, 5 features) and serializes it
2. **data_preprocessing_task** — Scales features using StandardScaler
3. **build_save_model_task** — Trains KMeans for k=2..10, finds optimal k using elbow method, saves model
4. **load_model_task** — Loads saved model and reports optimal number of clusters
5. **compare_models_task** ⭐ — Compares KMeans vs DBSCAN using silhouette score and picks the winner

## Results
```
MODEL COMPARISON REPORT
================================================
KMeans → Silhouette Score: 0.5283 | Clusters: 3
DBSCAN → Silhouette Score: -1.000 | Clusters: 1
------------------------------------------------
🏆 Winner: KMeans is better for this dataset!
================================================
```

## Project Structure
```
Airflow_Lab/
├── dags/
│   ├── data/
│   │   └── customers.csv       # Custom customer retail dataset
│   ├── model/                  # Saved model output
│   ├── src/
│   │   ├── __init__.py
│   │   └── lab.py              # ML logic (KMeans + DBSCAN)
│   └── airflow.py              # Airflow DAG definition (5 tasks)
├── docker-compose.yaml
├── .env
├── .gitignore
└── README.md
```

## How to Run
### Prerequisites
- Docker Desktop installed and running (min 4GB RAM allocated)

### Steps
```bash
# 1. Clone the repo
git clone https://github.com/CGunal7/Airflow_Lab.git
cd Airflow_Lab

# 2. Create .env file
echo "AIRFLOW_UID=$(id -u)" > .env

# 3. Initialize Airflow
docker compose up airflow-init

# 4. Start Airflow
docker compose up

# 5. Open browser → http://localhost:8080
# Login: airflow2 / airflow2
# Trigger the Airflow_Lab1 DAG

# 6. Stop when done
docker compose down
```

## Technologies Used
- **Apache Airflow 2.9.2** — Pipeline orchestration
- **Docker** — Containerized environment
- **scikit-learn** — KMeans, DBSCAN, silhouette score
- **kneed** — Elbow method for optimal cluster detection
- **pandas** — Data loading and preprocessing