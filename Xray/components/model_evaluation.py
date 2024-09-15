import sys
from typing import Tuple

# Importing necessary PyTorch modules for model training and evaluation
import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader

# Importing custom entities and modules from the Xray package
from Xray.entity.artifact_entity import (
    DataTransformationArtifact,  # Stores the results of data transformation
    ModelEvaluationArtifact,     # Stores the results of model evaluation
    ModelTrainerArtifact         # Stores the results of model training
)
from Xray.entity.config_entity import ModelEvaluationConfig  # Configurations related to model evaluation
from Xray.exception import XRayException  # Custom exception handling for the Xray project
from Xray.logger import logging  # Logging module for debugging and tracking information
from Xray.ml.model.arch import Net  # Model architecture (assumed to be a neural network class)

# This class handles model evaluation, including the evaluation process, configuration, and logging.
class ModelEvaluation:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        """
        Constructor to initialize the ModelEvaluation class with the required artifacts and configuration.

        Args:
            data_transformation_artifact (DataTransformationArtifact): Holds transformed test data
            model_evaluation_config (ModelEvaluationConfig): Configuration for the model evaluation
            model_trainer_artifact (ModelTrainerArtifact): Holds the path to the trained model
        """
        self.data_transformation_artifact = data_transformation_artifact  # Storing transformed test data
        self.model_evaluation_config = model_evaluation_config  # Storing the evaluation configuration
        self.model_trainer_artifact = model_trainer_artifact  # Storing the trained model artifact

    def configuration(self) -> Tuple[DataLoader, Module, float]:
        """
        This method sets up the model and its parameters for evaluation.

        Returns:
            A tuple containing the test dataloader, the model, and the loss function.
        """
        logging.info("Entered the configuration method of Model evaluation class")

        try:
            # Loading the transformed test dataset
            test_dataloader: DataLoader = (
                self.data_transformation_artifact.transformed_test_object
            )

            # Initializing the model architecture
            model: Module = Net()

            # Loading the pre-trained model using the path from ModelTrainerArtifact
            model: Module = torch.load(self.model_trainer_artifact.trained_model_path)

            # Sending the model to the specified device (GPU or CPU)
            model.to(self.model_evaluation_config.device)

            # Defining the loss function (Cross Entropy Loss for classification tasks)
            cost: Module = CrossEntropyLoss()

            # Putting the model in evaluation mode (disabling dropout, batch norm, etc.)
            model.eval()

            logging.info("Exited the configuration method of Model evaluation class")

            # Returning the test dataloader, the model, and the cost function
            return test_dataloader, model, cost

        except Exception as e:
            # Custom exception handling for the Xray project
            raise XRayException(e, sys)

    def test_net(self) -> float:
        """
        This method evaluates the model on the test dataset and calculates the accuracy and loss.

        Returns:
            The accuracy of the model on the test data.
        """
        logging.info("Entered the test_net method of Model evaluation class")

        try:
            # Retrieve test dataloader, model, and cost function from configuration method
            test_dataloader, net, cost = self.configuration()

            # Disabling gradient calculation for testing (improves performance)
            with torch.no_grad():
                holder = []  # To store the images, labels, and predictions for future analysis

                # Loop over the test data
                for _, data in enumerate(test_dataloader):
                    images = data[0].to(self.model_evaluation_config.device)  # Move images to device
                    labels = data[1].to(self.model_evaluation_config.device)  # Move labels to device

                    output = net(images)  # Forward pass through the model

                    loss = cost(output, labels)  # Compute loss between predictions and actual labels

                    # Get the predicted class labels (argmax gives the index of the highest score)
                    predictions = torch.argmax(output, 1)

                    # Store images, actual labels, and predictions in holder
                    for i in zip(images, labels, predictions):
                        h = list(i)
                        holder.append(h)

                    # Log actual labels, predictions, and the loss for debugging
                    logging.info(
                        f"Actual_Labels : {labels}     Predictions : {predictions}     Loss : {loss.item():.4f}"
                    )

                    # Accumulate loss over batches for computing average loss
                    self.model_evaluation_config.test_loss += loss.item()

                    # Accumulate correct predictions to calculate accuracy
                    self.model_evaluation_config.test_accuracy += (
                        (predictions == labels).sum().item()
                    )

                    # Update the number of batches processed
                    self.model_evaluation_config.total_batch += 1

                    # Update the total number of examples processed
                    self.model_evaluation_config.total += labels.size(0)

                    # Log the current loss and accuracy per batch for debugging
                    logging.info(
                        f"Model  -->   Loss : {self.model_evaluation_config.test_loss/ self.model_evaluation_config.total_batch} Accuracy : {(self.model_evaluation_config.test_accuracy / self.model_evaluation_config.total) * 100} %"
                    )

            # Final accuracy after processing the entire dataset
            accuracy = (
                self.model_evaluation_config.test_accuracy
                / self.model_evaluation_config.total
            ) * 100

            logging.info("Exited the test_net method of Model evaluation class")

            # Return the calculated accuracy
            return accuracy

        except Exception as e:
            # Custom exception handling in case of errors
            raise XRayException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        This method initiates the model evaluation and returns an artifact containing the evaluation results.

        Returns:
            ModelEvaluationArtifact: An artifact containing the accuracy of the model.
        """
        logging.info(
            "Entered the initiate_model_evaluation method of Model evaluation class"
        )

        try:
            # Perform the test and calculate accuracy
            accuracy = self.test_net()

            # Create a ModelEvaluationArtifact with the computed accuracy
            model_evaluation_artifact: ModelEvaluationArtifact = (
                ModelEvaluationArtifact(model_accuracy=accuracy)
            )

            logging.info(
                "Exited the initiate_model_evaluation method of Model evaluation class"
            )

            # Return the artifact containing evaluation results
            return model_evaluation_artifact

        except Exception as e:
            # Handle exceptions
            raise XRayException(e, sys)
