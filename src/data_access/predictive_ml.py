import sys
import pandas as pd
import numpy as np
from typing import Optional

from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME
from src.exception import MyException

class predictive_ml:

    def __init__(self) -> None:
        try:
            self.mongo_client = MongoDBClient(database_name = DATABASE_NAME)
        except Exception as e:
            raise MyException(e, sys)
    
    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        try:
            # Access specified collection from the default or the specified database
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]

            # Convert collection data to DataFrame and preprocess
            print(f'Fetching data from MongoDB')
            df = pd.DataFrame(list(collection.find()))
            print(f'Data fetched with len: {len(df)}')
            if ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'] in df.columns.to_list():
                df = df.drop(columns = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
            df.replace({'na' : np.nan}, inplace = True)
            return df
        except Exception as e:
            raise MyException(e, sys)