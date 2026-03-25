import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from kneed import KneeLocator


def load_data():
    data = pd.read_csv('/opt/airflow/dags/data/customers.csv')
    data = data.drop(columns=['customer_id'])
    print(f"✅ Data loaded: {data.shape[0]} rows, {data.shape[1]} features")
    print(data.describe())
    return pickle.dumps(data)


def data_preprocessing(data):
    df = pickle.loads(data)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    print(f"✅ Preprocessing done. Shape: {scaled.shape}")
    return pickle.dumps(scaled)


def build_save_model(data, filename):
    scaled_data = pickle.loads(data)
    sse = {}
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        sse[k] = kmeans.inertia_

    kneedle = KneeLocator(
        list(sse.keys()), list(sse.values()),
        curve='convex', direction='decreasing'
    )
    optimal_k = kneedle.knee if kneedle.knee else 3

    best_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    best_model.fit(scaled_data)
    model_path = f'/opt/airflow/dags/model/{filename}'
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    print(f"✅ KMeans model saved with k={optimal_k}")
    return pickle.dumps(sse)


def load_model_elbow(filename, sse):
    sse_dict = pickle.loads(sse)
    model_path = f'/opt/airflow/dags/model/{filename}'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    kneedle = KneeLocator(
        list(sse_dict.keys()), list(sse_dict.values()),
        curve='convex', direction='decreasing'
    )
    optimal_k = kneedle.knee if kneedle.knee else 3
    print(f"✅ KMeans optimal clusters (elbow method): {optimal_k}")
    print(f"✅ Model: {model}")
    return optimal_k


def compare_with_dbscan(data):
    scaled_data = pickle.loads(data)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(scaled_data)
    kmeans_score = silhouette_score(scaled_data, kmeans_labels)

    dbscan = DBSCAN(eps=0.8, min_samples=3)
    dbscan_labels = dbscan.fit_predict(scaled_data)
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

    if n_clusters_dbscan >= 2:
        dbscan_score = silhouette_score(scaled_data, dbscan_labels)
    else:
        dbscan_score = -1
        print("⚠️  DBSCAN found fewer than 2 clusters")

    print("=" * 50)
    print("       MODEL COMPARISON REPORT")
    print("=" * 50)
    print(f"  KMeans → Silhouette Score: {kmeans_score:.4f} | Clusters: 3")
    print(f"  DBSCAN → Silhouette Score: {dbscan_score:.4f} | Clusters: {n_clusters_dbscan}")
    print("-" * 50)
    if kmeans_score >= dbscan_score:
        print("  🏆 Winner: KMeans is better for this dataset!")
    else:
        print("  🏆 Winner: DBSCAN is better for this dataset!")
    print("=" * 50)

    return {"kmeans": kmeans_score, "dbscan": dbscan_score}