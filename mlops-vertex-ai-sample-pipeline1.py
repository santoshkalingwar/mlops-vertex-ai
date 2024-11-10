import kfp
from kfp import dsl
from google.cloud import storage
import pandas as pd
from google.cloud import aiplatform
from google_cloud_pipeline_components.v1.dataset import ImageDatasetCreateOp
from google_cloud_pipeline_components.v1.automl.training_job import AutoMLImageTrainingJobRunOp

# Constants for reduced resource allocation
BUDGET_MILLI_NODE_HOURS = 1000  # Reduced budget for smaller jobs
PIPELINE_ROOT = 'gs://vertex-ai-poc'  # Path where pipeline artifacts will be stored

# 1. Download File from GCS Component using @dsl.component
@dsl.component
def download_file_from_gcs(bucket_name: str, file_name: str, output_path: str):
    # Create a client to interact with GCS
    client = storage.Client()

    # Get the GCS bucket and blob (file)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Download the file to the specified output path
    blob.download_to_filename(output_path)
    print(f"File {file_name} downloaded from GCS to {output_path}.")

# 2. Process CSV Component
@dsl.component
def process_csv(input_path: str, output_path: str):
    # Read the CSV file using pandas
    df = pd.read_csv(input_path)

    # Example processing: Convert all column names to lowercase
    df.columns = df.columns.str.lower()

    # Save the processed DataFrame to the output path
    df.to_csv(output_path, index=False)
    print(f"Processed CSV saved to {output_path}.")

# 3. Upload File to GCS Component
@dsl.component
def upload_file_to_gcs(bucket_name: str, file_name: str, input_path: str):
    # Create a client to interact with GCS
    client = storage.Client()

    # Get the GCS bucket
    bucket = client.get_bucket(bucket_name)

    # Upload the file
    blob = bucket.blob(file_name)
    blob.upload_from_filename(input_path)
    print(f"File {file_name} uploaded to GCS from {input_path}.")

# 4. Define the Pipeline
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

    # Step 4: Create Dataset in Vertex AI (if not already created)
    dataset_create_op = ImageDatasetCreateOp(
        project='mlops-practice',
        display_name="employee-image-dataset",
        gcs_source="gs://bdi-vertex-ai-poc/employee_data.csv",  # GCS path to the CSV file
        import_schema_uri=aiplatform.schema.dataset.ioformat.image.single_label_classification,
    )

    # Step 5: AutoML training job with reduced budget (use the dataset object)
    training_job_run_op = AutoMLImageTrainingJobRunOp(
        project='mlops-practice',
        display_name="train-employee-data-model",
        prediction_type="classification",  # Set prediction type based on your use case
        model_type="CLOUD",  # Use CLOUD model for AutoML
        dataset=dataset_create_op.outputs["dataset"],  # Use the dataset object here
        model_display_name="employee-data-classification-model",
        training_fraction_split=0.6,
        validation_fraction_split=0.2,
        test_fraction_split=0.2,
        budget_milli_node_hours=BUDGET_MILLI_NODE_HOURS,  # Set a lower budget to avoid resource overuse
    )

    # Ensure task dependencies are properly set
    process_task.after(download_task)
    upload_task.after(process_task)
    dataset_create_op.after(upload_task)  # Ensure dataset is created before training job
    training_job_run_op.after(dataset_create_op)  # Ensure training job runs after dataset is created

# Compile the pipeline into a YAML file
compiler = kfp.compiler.Compiler()
compiler.compile(pipeline, 'employee_data_pipeline_with_smaller_resources.yaml')

# Initialize the Vertex AI SDK
aiplatform.init(
    project='mlops-practice',
    location="us-central1"  # Specify your region
)

# Prepare the pipeline job with smaller resource configuration
job = aiplatform.PipelineJob(
    display_name="employee-data-processing-pipeline",
    template_path='employee_data_pipeline_with_smaller_resources.yaml',  # Path to the compiled pipeline YAML file
    pipeline_root=PIPELINE_ROOT,  # GCS path for pipeline artifacts
    parameter_values={
        'bucket_name': 'vertex-ai-poc',  # Specify the GCS bucket where files are stored
        'input_file': 'employee_data.csv',  # The input file in your GCS bucket
        'output_file': 'processed_employee_data.csv',  # The output file name
    }
)

# Submit the pipeline job
job.submit()

print(f"Pipeline job submitted successfully.")
