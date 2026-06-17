from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from src.exception import MyException
from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, MODEL_BUCKET_NAME, MODEL_BUCKET_MODEL_NAME
from src.logger import logging
from src.utils.main_utils import load_object
import sys
import pandas as pd
from typing import Optional
import os
from dataclasses import dataclass
from src.utils.main_utils import read_yaml_file

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self._schema_config = read_yaml_file(file_path = SCHEMA_FILE_PATH)

        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self):
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
            # Move from src/component -> project root
            PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
    
            model_path = os.path.join(
                PROJECT_ROOT,
                "model_bucket",
                "predictive_maintenance_model.pkl"
            )
    
            print("Model Path:", model_path)
            print("Exists:", os.path.exists(model_path))
    
            if os.path.exists(model_path):
                model = load_object(file_path=model_path)
                logging.info(f"Production model loaded from {model_path}")
                return model
    
            logging.info("No production model found in model_bucket")
            return None
    
        except Exception as e:
            raise MyException(e, sys)
        
    def _drop_column(self, df):
        '''Drop the 'UDI, Product ID, TWF, HDF, PWF, OSF, RNF' column if it exists.'''
        logging.info(f"Dropping the columns")
        drop_col = self._schema_config['drop_columns']
        existing_cols = [col for col in drop_col if col in df.columns]

        if existing_cols:
            df = df.drop(columns=existing_cols)

        return df
    
    def _map_type_column(self, df):
        '''Map Type Column to 0 for Low, 1 for Medium, 2 for High.'''
        logging.info(f"Mapping 'Type' column to binary values")
        df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H' : 2})
        return df
    
    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            logging.info("Test data loaded and now transforming it for prediction...")

            x = self._drop_column(x)
            x = self._map_type_column(x)
            
            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded/exists.")
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"F1_Score for this model: {trained_model_f1_score}")

            best_model_f1_score=None
            best_model = self.get_best_model()

            if best_model is not None:
                logging.info("Computing F1 score for production model")
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logging.info(f"Production Model F1 Score: "f"{best_model_f1_score}")
                logging.info(f"New Model F1 Score: " f"{trained_model_f1_score}")
                logging.info(f"F1_Score-Production Model: {best_model_f1_score}, F1_Score-New Trained Model: {trained_model_f1_score}")
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference=trained_model_f1_score - tmp_best_model_score
                                           )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted = evaluate_model_response.is_model_accepted,
                s3_model_path = s3_model_path,
                trained_model_path = self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy = evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e