import os
import sys
import shutil

from src.exception import MyException
from src.entity.artifact_entity import ModelPusherArtifact
from src.constants import (
    MODEL_BUCKET_NAME,
    MODEL_BUCKET_MODEL_NAME
)
from src.logger import logging


class ModelPusher:

    def __init__(
        self,
        model_evaluation_artifact,
        model_pusher_config,
        model_trainer_artifact
    ):

        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.model_trainer_artifact = model_trainer_artifact

    def initiate_model_pusher(self):
        try:
            logging.info("Entered initiate_model_pusher method")

            # Source model path
            source_model_path = (self.model_trainer_artifact.trained_model_file_path)

            logging.info(f"Source model path: {source_model_path}")

            # Create model bucket directory
            os.makedirs(MODEL_BUCKET_NAME,exist_ok=True)

            logging.info(f"Created model bucket directory: {MODEL_BUCKET_NAME}")

            # Destination model path
            destination_model_path = os.path.join(
                MODEL_BUCKET_NAME,
                MODEL_BUCKET_MODEL_NAME
            )

            logging.info(f"Destination model path: {destination_model_path}")

            # Copy model
            shutil.copy2(
                source_model_path,
                destination_model_path
            )

            logging.info(
                f"Model copied successfully from " f"{source_model_path} to " f"{destination_model_path}"
            )

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=MODEL_BUCKET_NAME,
                s3_model_path=destination_model_path
            )

            logging.info(f"Model Pusher Artifact: {model_pusher_artifact}")

            logging.info("Exited initiate_model_pusher method")

            return model_pusher_artifact

        except Exception as e:
            raise MyException(e, sys)