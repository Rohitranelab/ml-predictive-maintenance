import sys
import os
from src.entity.config_entity import PredictionMaintenanceConfig
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame
from src.constants import MODEL_BUCKET_NAME, MODEL_BUCKET_MODEL_NAME
from src.utils.main_utils import load_object

class PredictiveMaintenance:
    def __init__(self,
                 Air_temperature,
                 Process_temperature,
                 Rotational_speed,
                 Torque,
                 Tool_Wear,
                 TypeEncoder
                ):
        '''
        Predictive Maintenance Constructor
        Input: all features of the trained model for prediction
        '''
        try:
            self.Air_temperature = Air_temperature
            self.Process_temperature = Process_temperature
            self.Rotational_speed = Rotational_speed
            self.Torque = Torque
            self.Tool_Wear = Tool_Wear
            self.TypeEncoder = TypeEncoder
            
        except Exception as e:
            raise MyException(e, sys)
        
    def get_predictive_data_as_data_frame(self):
        '''
        This function returns a DataFrame
        '''
        try:
            predictive_input_data = self.get_predictive_data_as_dict()
            return DataFrame(predictive_input_data)
        
        except Exception as e:
            raise MyException(e, sys)
        
    def get_predictive_data_as_dict(self):
        '''
        This function returns a dictionary from PredictiveMaintenance class input
        '''
        logging.info(f'Entered get_predictive_data_as_dict method as PredictiveMaintenance class')

        try:
            input_data = {
                'Air temperature [K]': [self.Air_temperature],
                'Process temperature [K]': [self.Process_temperature],
                'Rotational speed [rpm]': [self.Rotational_speed],
                'Torque [Nm]': [self.Torque],
                'Tool wear [min]': [self.Tool_Wear],
                'Type': [self.TypeEncoder]
            }

            logging.info(f'Created predictive data dict')
            logging.info(f'Exited get_predictive_data_as_dict method as PredictiveMaintenance class')
            return input_data
        
        except Exception as e:
            raise MyException(e, sys)

class PredictionMaintenanceClassifier:
    def __init__(self, prediction_pipeline_config: PredictionMaintenanceConfig = PredictionMaintenanceConfig(), ) -> None:
        '''
        param prediction_pipeline_config: Configuration for prediction the value
        '''
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)
        
    def predict(self, dataframe) -> str:
        '''
        This is the method of PredictiveMaintenanceClassifier
        Returns: Prediction in string format
        '''
        try:
                logging.info("Entered predict method of PredictionMaintenanceClassifier")
                model_path = os.path.join(
                    MODEL_BUCKET_NAME,
                    MODEL_BUCKET_MODEL_NAME
                )
                logging.info(f"Loading model from: {model_path}")
                model = load_object(file_path=model_path)
                result = model.predict(dataframe)
                logging.info("Prediction completed successfully")
                return result

        except Exception as e:
            raise MyException(e, sys)