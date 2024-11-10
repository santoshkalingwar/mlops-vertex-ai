#!pip install kfp google-cloud-storage google-cloud-pipeline-components
#!pip install google-cloud-storage

import kfp
from kfp import dsl
from google.cloud import storage
import pandas as pd

# Custom component to download file from GCS
@dsl.component
def download_file_from_gcs(bucket_name: str, file_name: str, output_path: str):
    """Download a file from GCS bucket"""
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.download_to_filename(output_path)
    print(f"File {file_name} downloaded to {output_path}")

# Custom component to upload file to GCS
@dsl.component
def upload_file_to_gcs(bucket_name: str, file_name: str, input_path: str):
    """Upload a file to GCS bucket"""
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_filename(input_path)
    print(f"File {input_path} uploaded to gs://{bucket_name}/{file_name}")

# Custom component to process the CSV data
@dsl.component
def process_csv(input_path: str, output_path: str):
    """Process the CSV file (e.g., increase salary by 10%)"""
    df = pd.read_csv(input_path)
    # Example: Increase salary by 10%
    df['Salary'] = df['Salary'] * 1.1
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

# Define the pipeline
@dsl.pipeline(
    name="Employee Data Processing Pipeline",
    description="Download, process, and upload CSV data from/to GCS"
)
def pipeline(bucket_name: str, input_file: str, output_file: str):
    # Step 1: Download the file from GCS
    download_task = download_file_from_gcs(
        bucket_name=bucket_name,
        file_name=input_file,
        output_path='/tmp/input.csv'
    )

    # Step 2: Process the CSV file
    process_task = process_csv(
        input_path='/tmp/input.csv',
        output_path='/tmp/processed.csv'
    )

    # Step 3: Upload the processed file back to GCS
    upload_task = upload_file_to_gcs(
        bucket_name=bucket_name,
        file_name=output_file,
        input_path='/tmp/processed.csv'
    )

    # Set dependencies between tasks
    process_task.after(download_task)
    upload_task.after(process_task)

# Compile the pipeline into a YAML file
compiler = kfp.compiler.Compiler()
compiler.compile(pipeline, 'employee_data_pipeline.yaml')





import google.cloud.aiplatform as aip

# Set your project ID, region, and the path to the compiled pipeline YAML
project_id = 'mlops-practice'
pipeline_root_path = 'gs://vertex-ai-poc'  # GCS path where pipeline artifacts will be stored
pipeline_yaml_path = 'employee_data_pipeline.yaml'  # The path to the compiled pipeline YAML file
pipeline_display_name = 'employee-data-processing-pipeline'

# Set environment variable for authentication (if running locally or outside Vertex AI Workbench)
# Make sure GOOGLE_APPLICATION_CREDENTIALS is set to the service account key file
# import os
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'path_to_your_service_account_key.json'

# Initialize the Vertex AI SDK
aip.init(
    project=project_id,
    location="us-central1"  # Specify your region
)

# Prepare the pipeline job
job = aip.PipelineJob(
    display_name=pipeline_display_name,
    template_path=pipeline_yaml_path,  # Path to the pipeline YAML
    pipeline_root=pipeline_root_path,  # Root directory to store pipeline artifacts (GCS path)
    parameter_values={
        'bucket_name': 'vertex-ai-poc',            # Specify the GCS bucket where files are stored
        'input_file': 'employee_data.csv',  # The input file in your GCS bucket
        'output_file': 'processed_employee_data.csv',  # The output file name
    }
)

# Submit the pipeline job
job.submit()

print(f"Pipeline job '{pipeline_display_name}' submitted successfully.")
