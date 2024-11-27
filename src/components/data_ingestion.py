import os
import pandas as pd 
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class DataIngestionconfig:
    raw_data_path:str = 'artifacts/data.csv'


class DataIngestionProcess:
    def __init__(self):
        self.data_ingestion_config= DataIngestionconfig()

    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv(r'notebook\data\loan-prediction-dataset.csv')
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index = False, header = True)

          
            logging.info("Data ingestion has been completed...")
            return self.data_ingestion_config.raw_data_path
        except Exception as e:
            raise CustomException(e)

if __name__ == '__main__':
    model = DataIngestionProcess()
    raw_data_path = model.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data,_ = data_transformation.initiate_data_transformation(raw_data_path)

    model_trainer = ModelTrainer()
    classification_results, regression_results = model_trainer.initiate_model_trainer(data)

    # Print classification results
    print("Classification Results:")
    print(f"Report: {classification_results[0]}")
    print(f"Best Model: {classification_results[1]}")
    print(f"Best Model Test Score: {classification_results[2]}")

    # Print regression results
    print("\nRegression Results:")
    print(f"Report: {regression_results[0]}")
    print(f"Best Model: {regression_results[1]}")
    print(f"Best Model Test Score: {regression_results[2]}")