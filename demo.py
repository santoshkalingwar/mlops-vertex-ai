# pip install google-cloud-aiplatform kfp

from kfp import dsl
from kfp.v2.dsl import component
from google.cloud import aiplatform
import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

# Define a simple training component (using scikit-learn for simplicity)
@component
def train_model_component() -> str:
    # Load a simple dataset (iris dataset for demonstration)
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42)

    # Train a model (RandomForest for simplicity)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Save the model to a file
    model_file_path = '/tmp/iris_model.pkl'
    joblib.dump(model, model_file_path)

    # Return the path as a string
    return model_file_path


# Define a component to upload the model to Vertex AI
@component
def upload_model_to_vertex_ai(model_path: str):
    # Initialize the Vertex AI client
    aiplatform.init(project="YOUR_PROJECT_ID", location="us-central1")

    # Upload the model to Vertex AI as a string (path)
    model = aiplatform.Model.upload(
        display_name="iris-model",
        artifact_uri=model_path,  # Ensure that `model_path` is a string
        serving_container_image_uri="gcr.io/cloud-aiplatform/training/tf2-cpu.2-3:latest",
    )

    # Deploy model to an endpoint
    model.deploy(machine_type="n1-standard-2")
    return model.resource_name


# Define the pipeline function
@dsl.pipeline(
    name="vertex-ai-pipeline",
    pipeline_root="gs://YOUR_BUCKET_NAME/pipeline_root"
)
def vertex_ai_pipeline():
    # Step 1: Train the model
    model_path = train_model_component()

    # Step 2: Upload and deploy model to Vertex AI
    upload_model_to_vertex_ai(model_path=model_path)  # Pass the path as a string


# Compile the pipeline
from kfp.v2.compiler import Compiler

Compiler().compile(
    pipeline_func=vertex_ai_pipeline,
    package_path="vertex_ai_pipeline.json"
)




from google.cloud import aiplatform

aiplatform.init(project="YOUR_PROJECT_ID", location="us-central1")

# Define the pipeline job
pipeline_job = aiplatform.PipelineJob(
    display_name="vertex-ai-pipeline-job",
    template_path="vertex_ai_pipeline.json",
    enable_caching=False
)

# Run the pipeline job
pipeline_job.run()
