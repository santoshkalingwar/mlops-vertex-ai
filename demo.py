#pip install google-cloud-aiplatform kfp xgboost pandas scikit-learn

import os
import joblib
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from kfp import dsl
from kfp.v2.dsl import component
from google.cloud import aiplatform

# Simulate loading and preprocessing data (e.g., CSV to dataframe)
@component
def preprocess_data_component() -> str:
    """Simulate loading and preprocessing the dataset."""
    # Create a simple dataset
    data = {
        "feature1": [1.0, 2.1, 3.2, 4.3, 5.4],
        "feature2": [10.1, 20.2, 30.3, 40.4, 50.5],
        "target": [0, 1, 0, 1, 0]
    }
    
    df = pd.DataFrame(data)
    
    # Preprocess data (simple feature engineering)
    X = df[['feature1', 'feature2']]
    y = df['target']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save preprocessed data to a file for later use
    preprocessed_data_path = '/tmp/preprocessed_data.pkl'
    joblib.dump((X_train, X_test, y_train, y_test), preprocessed_data_path)
    
    return preprocessed_data_path

# Train an XGBoost model
@component
def train_model_component(preprocessed_data_path: str) -> str:
    """Train an XGBoost model."""
    # Load preprocessed data
    X_train, X_test, y_train, y_test = joblib.load(preprocessed_data_path)
    
    # Train an XGBoost model
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3)
    model.fit(X_train, y_train)
    
    # Save the trained model
    model_file_path = '/tmp/xgboost_model.pkl'
    joblib.dump(model, model_file_path)
    
    return model_file_path

# Upload and deploy the model to Vertex AI
@component
def upload_and_deploy_model(model_path: str):
    """Upload the trained model to Vertex AI and deploy it."""
    # Initialize the Vertex AI client
    aiplatform.init(project="YOUR_PROJECT_ID", location="us-central1")
    
    # Upload the model to Vertex AI
    model = aiplatform.Model.upload(
        display_name="xgboost-model",
        artifact_uri=f"gs://YOUR_BUCKET_NAME/{os.path.basename(model_path)}",  # Upload to GCS
        serving_container_image_uri="gcr.io/cloud-aiplatform/training/tf2-cpu.2-3:latest",
    )
    
    # Deploy the model
    deployed_model = model.deploy(machine_type="n1-standard-2")
    
    return deployed_model.resource_name

# Define the pipeline
@dsl.pipeline(
    name="vertex-ai-xgboost-pipeline",
    pipeline_root="gs://YOUR_BUCKET_NAME/pipeline_root"  # Your GCS bucket path
)
def vertex_ai_pipeline():
    # Step 1: Preprocess the data
    preprocessed_data_path = preprocess_data_component()
    
    # Step 2: Train the model
    model_path = train_model_component(preprocessed_data_path=preprocessed_data_path)
    
    # Step 3: Upload and deploy the model to Vertex AI
    upload_and_deploy_model(model_path=model_path)

# Compile the pipeline
from kfp.v2.compiler import Compiler

Compiler().compile(
    pipeline_func=vertex_ai_pipeline,
    package_path="vertex_ai_xgboost_pipeline.json"
)


from google.cloud import aiplatform

# Initialize the Vertex AI SDK
aiplatform.init(project="YOUR_PROJECT_ID", location="us-central1")

# Define the pipeline job
pipeline_job = aiplatform.PipelineJob(
    display_name="vertex-ai-xgboost-pipeline-job",
    template_path="vertex_ai_xgboost_pipeline.json",  # Path to compiled pipeline JSON file
    enable_caching=False  # Disable caching to always run the pipeline from scratch
)

# Run the pipeline job
pipeline_job.run()
