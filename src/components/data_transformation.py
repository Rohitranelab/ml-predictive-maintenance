import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import SCHEMA_FILE_PATH, TARGET_COLUMN, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file

class DataTransforamtion:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path = SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)
    
    @staticmethod
    def read_data(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
    
    def get_data_transformer_object(self) -> Pipeline:
        logging.info(f"Entered get_data_transformer_object method of DataTransformation class")

        try:
            # initialize transformers
            numeric_transformer = StandardScaler()
            logging.info(f"Transformers Initialized: StandardScaler")

            # Load schema configuration
            num_features = self._schema_config['num_features']

            # Creating preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers = [
                    ('StandardScaler', numeric_transformer, num_features)
                ],
                remainder = "passthrough"
            )

            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps = [('preprocessor', preprocessor)])
            logging.info(f"Final Pipeline Ready!!!")
            logging.info(f"Exited get_data_transformer_object_method of DataTransformation class")
            return final_pipeline

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
    
    def initiate_data_transform(self) -> DataTransformationArtifact:
        '''
        Initiates the data transformation component for the pipeline.
        '''
        try:
            logging.info('Data Transformation Started!!!!')
            if not self.data_validation_artifact.validation_status:
                raise MyException(self.data_validation_artifact.message)
            
            # Load train and test data
            train_df = self.read_data(file_path = self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path= self.data_ingestion_artifact.test_file_path)
            logging.info('Train-Test data loaded')

            input_feature_train_df = train_df.drop(columns = [TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns = [TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info(f'Input and Target cols defined for both train and test df.')

            # Apply custom transformations in specified sequence
            input_feature_train_df = self._drop_column(input_feature_train_df)
            input_feature_train_df = self._map_type_column(input_feature_train_df)

            input_feature_test_df = self._drop_column(input_feature_test_df)
            input_feature_test_df = self._map_type_column(input_feature_test_df)

            logging.info(f'Custom transformations applied to train and test data')

            logging.info('Starting data transformation')
            preprocessor = self.get_data_transformer_object()
            logging.info(f'Got the preprocessor object')

            logging.info(f'Initializing transformation for Training-data')
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info(f'Initializing transformation for Testing-data')
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info(f'Transformation done end to end to train-test df.')

            logging.info(f'Applying SMOTEENN for handling imbalanced dataset.')
            smt = SMOTEENN(sampling_strategy = "minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )
            logging.info(f'SMOTEENN applied to train-test df.')

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info(f'feature-target concatentation done for train-test df.')

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformation_train_file_path, array = train_arr)
            save_numpy_array_data(self.data_transformation_config.transformation_test_file_path, array = test_arr)
            logging.info(f'Saving transformation object and tranformed files.')

            logging.info(f'Data Transforamtion completed successfully')
            return DataTransformationArtifact(
                transformed_object_file_path = self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path = self.data_transformation_config.transformation_train_file_path,
                transformed_test_file_path = self.data_transformation_config.transformation_test_file_path
            )
        
        except Exception as e:
            raise MyException(e, sys) from e