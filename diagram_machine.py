from diagrams import Diagram, Cluster
from diagrams.aws.compute import EC2
from diagrams.aws.ml import SagemakerModel
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.client import Client
from diagrams.programming.flowchart import PredefinedProcess, InputOutput

with Diagram("Model Training Pipeline", show=False):
    # Data preparation
    with Cluster("Data Preparation"):
        load_data = InputOutput("Load CSV Data")
        process_image_path = PredefinedProcess("Process Image Paths")
        clean_angular_velocity = PredefinedProcess("Clean Angular Velocity")
        load_data >> process_image_path >> clean_angular_velocity
    
    # Image processing
    with Cluster("Image Processing"):
        load_images = PredefinedProcess("Load Images")
        preprocess_images = PredefinedProcess("Preprocess Images")
        split_data = PredefinedProcess("Train-Validation Split")
        clean_angular_velocity >> load_images >> preprocess_images >> split_data
    
    # Model architecture
    with Cluster("NVIDIA Model"):
        conv_layers = [SagemakerModel("Conv2D Layer 1"),
                       SagemakerModel("Conv2D Layer 2"),
                       SagemakerModel("Conv2D Layer 3"),
                       SagemakerModel("Conv2D Layer 4"),
                       SagemakerModel("Conv2D Layer 5")]
        dense_layers = [SagemakerModel("Dense Layer 1"),
                        SagemakerModel("Dense Layer 2"),
                        SagemakerModel("Dense Layer 3"),
                        SagemakerModel("Output Layer")]

    # Connect conv_layers individually
    prev_layer = split_data
    for layer in conv_layers:
        prev_layer >> layer
        prev_layer = layer

    # Connect dense_layers individually
    for layer in dense_layers:
        prev_layer >> layer
        prev_layer = layer
    
    # Training and evaluation
    training = PredefinedProcess("Training")
    evaluation = PredefinedProcess("Evaluation (MSE, MAE)")

    prev_layer >> training >> evaluation

