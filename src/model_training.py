import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.exception import CustomException
from config.path_config import *
from config.model_params import *
from utils.common_function import read_yaml, load_data
from scipy.stats import uniform, randint
import mlflow
import mlflow.sklearn 

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        self.params_dist = LIGHTGM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):
        """Load data from CSV and split into features and target."""
        try:
            logger.info("Loading training and testing data")
            train_df = load_data(self.train_path)
            logger.info(f"Training data shape: {train_df.shape}")
            test_df = load_data(self.test_path)
            logger.info(f"Testing data shape: {test_df.shape}")
            X_train = train_df.drop(columns = ['booking_status'])
            y_train = train_df['booking_status']
            X_test = test_df.drop(columns = ['booking_status'])
            y_test = test_df['booking_status']
            logger.info("Data loaded and split into features and target")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Error loading and splitting data: {e}")
            raise CustomException("failed to load and split the data", sys)
    
    def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Starting model training with LightGBM")
            lgbm_model = LGBMClassifier(random_state=self.random_search_params["random_state"])
            logger.info("Performing Randomized Search for hyperparameter tuning")
            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params["n_iter"],
                scoring=self.random_search_params["scoring"],
                cv=self.random_search_params["cv"],
                verbose=self.random_search_params["verbose"],
                random_state=self.random_search_params["random_state"],
                n_jobs=-1
            )
            
            logger.info("starting our model training")
            random_search.fit(X_train, y_train)
            logger.info("hyperparameter tuning completed")
            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_
            logger.info(f"Best hyperparameters: {best_params}")
            return best_lgbm_model
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise CustomException("failed to train the model", sys)

    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating the model on test data")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            logger.info(f"Model Evaluation Metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise CustomException("failed to evaluate the model", sys)
    
    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            logger.info(f"Saving the model to {self.model_output_path}")
            joblib.dump(model, self.model_output_path)
            logger.info(f"Model saved successfully at {self.model_output_path}")
        except Exception as e:
            logger.error(f"Error saving the model: {e}")
            raise CustomException("failed to save the model", sys)
        
    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting our model training process")
                logger.info("Startng our mlflow logging")
                logger.info("Logging the training and testing data to mlflow")
                mlflow.log_artifact(self.train_path, artifact_path="datasets")
                mlflow.log_artifact(self.test_path, artifact_path="datasets")
                X_train, y_train, X_test, y_test = self.load_and_split_data()
                best_lgbm_model = self.train_lgbm(X_train, y_train)
                metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)
                self.save_model(best_lgbm_model)
                mlflow.log_artifact(self.model_output_path, artifact_path="model")
                logger.info("Logging model parameters and metrics to mlflow")
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)
                logger.info("Model training process completed successfully")
        except Exception as e:
            logger.error(f"Error in model training process: {e}")
            raise CustomException("Error in the model training process", sys)
        
if __name__ == "__main__":
    trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
    trainer.run()