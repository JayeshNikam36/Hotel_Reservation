import os
import sys
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.exception import CustomException
from config.path_config import *
from utils.common_function import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)


class DataPreprocessor:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def preprocess_data(self, df):
        """Drop unnecessary columns, encode categorical features, and handle skewness."""
        try:
            logger.info("Starting data preprocessing...")

            # Drop ID column and duplicates
            logger.info("Dropping unnecessary columns")
            df.drop(columns=["Booking_ID"], inplace=True, errors="ignore")
            df.drop_duplicates(inplace=True)

            # Columns from config
            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]

            # Encode categorical columns
            logger.info("Applying Label Encoding")
            label_encoder = LabelEncoder()
            mappings = {}

            for col in cat_cols:
                if col in df.columns:
                    df[col] = label_encoder.fit_transform(df[col])
                    mappings[col] = {
                        label: code
                        for label, code in zip(
                            label_encoder.classes_,
                            label_encoder.transform(label_encoder.classes_),
                        )
                    }
                else:
                    logger.warning(f"Column {col} not found in dataframe.")

            logger.info("Label mappings are:")
            for col, mapping in mappings.items():
                logger.info(f"{col}: {mapping}")

            # Handle skewness
            logger.info("Handling skewness in numerical columns")
            skewness_threshold = self.config["data_processing"]["skewness_threshold"]
            skewness = df[num_cols].apply(lambda x: x.skew())

            for col in skewness[skewness > skewness_threshold].index:
                df[col] = np.log1p(df[col])

            return df

        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            raise CustomException("failed in data preprocessing", sys)

    def balance_data(self, df):
        """Balance dataset using SMOTE."""
        try:
            logger.info("Handling imbalanced data")
            X = df.drop(columns=["booking_status"])
            y = df["booking_status"]

            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)

            balanced_df = pd.DataFrame(X_res, columns=X.columns)
            balanced_df["booking_status"] = y_res

            logger.info("Data balancing completed")
            return balanced_df

        except Exception as e:
            logger.error(f"Error in balancing data: {e}")
            raise CustomException("failed in balancing the data", sys)

    def select_features(self, df):
        """Select top features using RandomForest importance."""
        try:
            logger.info("Starting feature selection")
            X = df.drop(columns="booking_status")
            y = df["booking_status"]

            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)

            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame(
                {"feature": X.columns, "importance": feature_importance}
            )

            # Sort by importance
            top_features_importance_df = feature_importance_df.sort_values(
                by="importance", ascending=False
            )

            number_features_to_select = self.config["data_processing"]["number_of_features"]
            top_features = top_features_importance_df["feature"].head(
                number_features_to_select
            ).values

            # Keep only top features + target column
            top_df = df[top_features.tolist() + ["booking_status"]]
            logger.info("Feature selection completed successfully")
            return top_df

        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            raise CustomException("failed in feature selection", sys)

    def save_data(self, df, file_path):
        """Save dataframe to CSV."""
        try:
            logger.info(f"Saving processed data to {file_path}")
            df.to_csv(file_path, index=False)
            logger.info("Data saved successfully")

        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {e}")
            raise CustomException("failed to save the data", sys)

    def process(self):
        """Run the complete preprocessing pipeline."""
        try:
            logger.info("Loading data from raw directory")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            logger.info("Preprocessing training data")
            train_df = self.preprocess_data(train_df)

            logger.info("Preprocessing testing data")
            test_df = self.preprocess_data(test_df)

            logger.info("Balancing training data")
            train_df = self.balance_data(train_df)

            logger.info("Balancing testing data")
            test_df = self.balance_data(test_df)

            logger.info("Selecting features from training data")
            train_df = self.select_features(train_df)

            logger.info("Selecting features from testing data")
            test_df = test_df[train_df.columns]

            # Save processed datasets
            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data preprocessing completed successfully")

        except Exception as e:
            logger.error(f"Error in data processing pipeline: {e}")
            raise CustomException("failed in data processing", sys)


if __name__ == "__main__":
    processor = DataPreprocessor(
        TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH
    )
    processor.process()
