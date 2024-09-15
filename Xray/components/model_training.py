import os
import sys

import bentoml
import joblib
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from tqdm import tqdm

from Xray.constant.training_pipeline import *
from Xray.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from Xray.entity.config_entity import ModelTrainerConfig
from Xray.exception import XRayException
from Xray.logger import logging
from Xray.ml.model.arch import Net




class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        """
        Initializes the ModelTrainer class with configuration and data artifacts.
        
        :param data_transformation_artifact: Holds transformed training and testing datasets.
        :param model_trainer_config: Configuration that includes model parameters, device settings, and paths.
        """
        self.model_trainer_config: ModelTrainerConfig = model_trainer_config

        self.data_transformation_artifact: DataTransformationArtifact = (
            data_transformation_artifact
        )

        # Initialize the model architecture from the Net class (defined in another file)
        self.model: Module = Net()



    # This method is responsible for training the model for one epoch using stochastic gradient descent (SGD) optimization.
    def train(self, optimizer: Optimizer) -> None:
        """
        Trains the model for one epoch.

        :param optimizer: The optimizer that updates the model's parameters.
        """
        logging.info("Entered the train method of Model trainer class")

        try:
            # Set the model to training mode (important for layers like dropout and batch normalization)
            self.model.train()

            # Initialize a progress bar for the training loop
            pbar = tqdm(self.data_transformation_artifact.transformed_train_object)

            correct: int = 0  # Track the number of correct predictions
            processed: int = 0  # Track the number of processed examples

            # Loop through the batches in the training dataset
            for batch_idx, (data, target) in enumerate(pbar):
                # Move the input data and target labels to the specified device (CPU or GPU)
                data, target = data.to(DEVICE), target.to(DEVICE)

                # Zero the gradients to avoid accumulation (which happens by default in PyTorch)
                optimizer.zero_grad()

                # Forward pass: make predictions based on the input data
                y_pred = self.model(data)

                # Calculate the loss using negative log-likelihood (nll_loss)
                loss = F.nll_loss(y_pred, target)

                # Backpropagate the loss to compute gradients
                loss.backward()

                # Update the model parameters based on the computed gradients
                optimizer.step()

                # Get the predicted class index (the index with the highest log-probability)
                pred = y_pred.argmax(dim=1, keepdim=True)

                # Count the number of correct predictions
                correct += pred.eq(target.view_as(pred)).sum().item()

                # Update the processed data count
                processed += len(data)

                # Update the progress bar with current loss and accuracy
                pbar.set_description(
                    desc=f"Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}"
                )

            logging.info("Exited the train method of Model trainer class")

        except Exception as e:
            raise XRayException(e, sys)
        



    def test(self) -> None:
        """
        Evaluates the model on the test dataset and calculates loss and accuracy.
        """
        try:
            logging.info("Entered the test method of Model trainer class")

            # Set the model to evaluation mode (important for layers like dropout and batch normalization)
            self.model.eval()

            test_loss: float = 0.0  # Initialize the total test loss
            correct: int = 0  # Track the number of correct predictions

            # Disable gradient calculations for testing (saves memory and computation)
            with torch.no_grad():
                for data, target in self.data_transformation_artifact.transformed_test_object:
                    # Move the data and target to the specified device (CPU or GPU)
                    data, target = data.to(DEVICE), target.to(DEVICE)

                    # Forward pass: make predictions on the test data
                    output = self.model(data)

                    # Compute the test loss
                    test_loss += F.nll_loss(output, target, reduction="sum").item()

                    # Get the predicted class index
                    pred = output.argmax(dim=1, keepdim=True)

                    # Count the number of correct predictions
                    correct += pred.eq(target.view_as(pred)).sum().item()

                # Calculate the average loss
                test_loss /= len(self.data_transformation_artifact.transformed_test_object.dataset)

                print(
                    "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                        test_loss,
                        correct,
                        len(self.data_transformation_artifact.transformed_test_object.dataset),
                        100.0 * correct / len(self.data_transformation_artifact.transformed_test_object.dataset),
                    )
                )

            logging.info("Exited the test method of Model trainer class")

        except Exception as e:
            raise XRayException(e, sys)

        
        

        

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info(
                "Entered the initiate_model_trainer method of Model trainer class"
            )

            model: Module = self.model.to(self.model_trainer_config.device)

            optimizer: Optimizer = torch.optim.SGD(
                model.parameters(), **self.model_trainer_config.optimizer_params
            )

            scheduler: _LRScheduler = StepLR(
                optimizer=optimizer, **self.model_trainer_config.scheduler_params
            )

            for epoch in range(1, self.model_trainer_config.epochs + 1):
                print("Epoch : ", epoch)

                self.train(optimizer=optimizer)

                optimizer.step()

                scheduler.step()

                self.test()

            os.makedirs(self.model_trainer_config.artifact_dir, exist_ok=True)

            torch.save(model, self.model_trainer_config.trained_model_path)

            train_transforms_obj = joblib.load(
                self.data_transformation_artifact.train_transform_file_path
            )

            bentoml.pytorch.save_model(
                name=self.model_trainer_config.trained_bentoml_model_name,
                model=model,
                custom_objects={
                    self.model_trainer_config.train_transforms_key: train_transforms_obj
                },
            )

            model_trainer_artifact: ModelTrainerArtifact = ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model_path
            )

            logging.info(
                "Exited the initiate_model_trainer method of Model trainer class"
            )

            return model_trainer_artifact

        except Exception as e:
            raise XRayException(e, sys)