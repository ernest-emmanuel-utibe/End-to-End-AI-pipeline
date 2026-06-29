import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# 1. Dynamically configure connection to the central MLflow Tracking mesh service
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Production_End_To_End_Pipeline")

class EndToEndAIPipeline:
    def __init__(self):
        self.model_name = "Core_Production_Classifier"
        self.email_regex = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        
    def ingest_and_version_data(self) -> pd.DataFrame:
        """Stage 1: Ingest and mock a raw data stream with PII risks."""
        print("[INFO] Stage 1: Ingesting and versioning data stream...")
        np.random.seed(42)
        X = np.random.rand(200, 4)
        
        # Inject raw string email addresses to simulate data leakage vulnerability
        emails = [f"user_{i}@leakdomain.com" if i % 20 == 0 else "clean_record" for i in range(200)]
        y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
        
        df = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3', 'feature_4'])
        df['user_metadata'] = emails
        df['target'] = y
        return df

    def secure_and_sanitize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage 2: Filter inputs via safety gateway to block data leakage."""
        print("[INFO] Stage 2: Running security gateway scrubbing loops...")
        df['user_metadata'] = df['user_metadata'].apply(
            lambda x: re.sub(self.email_regex, "[REDACTED_PII]", str(x))
        )
        # Drop unhashed text entirely to protect private system integrity
        return df.drop(columns=['user_metadata'])

    def train_and_track_model(self, df: pd.DataFrame):
        """Stage 3 & 4: Execute model training and push tracked metrics to server."""
        print("[INFO] Stage 3 & 4: Executing tracked model optimization run...")
        X = df.drop(columns=['target'])
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        with mlflow.start_run() as run:
            n_estimators = 120
            learning_rate = 0.05
            
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("learning_rate", learning_rate)
            
            # Train model
            model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
            model.fit(X_train, y_train)
            
            # Metric validation extraction
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            mlflow.log_metric("test_accuracy", accuracy)
            print(f"[SUCCESS] Training run finished. Metric Score: {accuracy:.4f}")
            
            # Stage 5: Evaluation performance checkpoint barrier
            if self.evaluate_performance_gate(accuracy):
                print("[INFO] Registering approved model artifact into global engine...")
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model_artifact",
                    registered_model_name=self.model_name
                )
                return run.info.run_id
            else:
                print("[ALERT] Performance SLA gate check failed. Registry update blocked.")
                return None

    def evaluate_performance_gate(self, current_accuracy: float) -> bool:
        """Stage 5: Systematic output gate validation rules."""
        print("[INFO] Stage 5: Auditing metrics against production SLAs...")
        MINIMUM_ACCURACY_THRESHOLD = 0.80
        return current_accuracy >= MINIMUM_ACCURACY_THRESHOLD

if __name__ == "__main__":
    print("--- STARTING WORKER TASK PIPELINE EXECUTION ---")
    pipeline = EndToEndAIPipeline()
    
    raw_data = pipeline.ingest_and_version_data()
    clean_data = pipeline.secure_and_sanitize_data(raw_data)
    run_id = pipeline.train_and_track_model(clean_data)
    
    if run_id:
        print(f"--- PIPELINE EXECUTION SUCCESSFUL (Run ID: {run_id}) ---")
    else:
        print("--- PIPELINE RUN TERMINATED BY AUDIT GATE ---")
