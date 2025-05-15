import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from typing import Tuple, Dict
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        """Initialize the DataPreprocessor with necessary encoders and scalers."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.numerical_columns = ['age', 'study_hours_per_day', 'social_media_hours',
                                 'netflix_hours', 'attendance_percentage', 
                                 'sleep_hours','exercise_frequency',
                                 'mental_health_rating','exam_score']
        
        self.categorical_columns = ['gender','part_time_job','diet_quality',
                                    'parental_education_level','internet_quality',
                                    'extracurricular_participation']
    def check_missing_values(self, df: pd.DataFrame) -> None:
        """Check for missing values in the dataset."""
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning("Missing values found:")
            for column, count in missing_values[missing_values > 0].items():
                logger.warning(f"{column}: {count} missing values")
        else:
            logger.info("No missing values found in the dataset")
            
    def check_duplicates(self, df: pd.DataFrame) -> None:
        """Check for duplicate entries in the dataset."""
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate entries")
        else:
            logger.info("No duplicate entries found")
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df['parental_education_level'] = df['parental_education_level'].fillna(df['parental_education_level'].mode()[0])
        return df
            
    def create_study_efficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create study efficiency feature."""
        df['study_efficiency'] = df['study_hours_per_day'] / (df['social_media_hours'] + df['netflix_hours'] + 1)
        return df
        
    def create_lifestyle_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lifestyle score feature."""
        df['lifestyle_score'] = (
            df['sleep_hours'] / 8 +  # Normalize sleep hours (8 hours is ideal)
            df['exercise_frequency'] / 7 +  # Normalize exercise frequency
            df['mental_health_rating'] / 10  # Mental health is already on 1-10 scale
        ) / 3  # Average the scores
        return df
        
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables using Label Encoding."""
        categorical_columns = ['gender', 'diet_quality', 'parental_education_level', 
                             'internet_quality']
        
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
            df[column] = self.label_encoders[column].fit_transform(df[column])
            
        # Convert binary variables to numeric
        df['part_time_job'] = (df['part_time_job'] == 'Yes').astype(int)
        df['extracurricular_participation'] = (df['extracurricular_participation'] == 'Yes').astype(int)
        
        return df
        
    def scale_numerical_features(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Scale numerical features using StandardScaler."""
        df[columns] = self.scaler.fit_transform(df[columns])
        return df
        
    def preprocess_data(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the data with all necessary steps.
        
        Args:
            df: Input DataFrame
            is_training: Whether this is training data or prediction data
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        try:
            logger.info("Starting data preprocessing...")
            
            # Make a copy to avoid modifying original data
            df_processed = df.copy()
            
            # Check data quality
            self.check_missing_values(df_processed)
            self.check_duplicates(df_processed)
            self.handle_missing_values(df_processed)
            
            # Handle outliers in numerical columns
            numerical_columns = ['age', 'study_hours_per_day', 'social_media_hours', 
                               'netflix_hours', 'attendance_percentage', 'sleep_hours',
                               'exercise_frequency', 'mental_health_rating']
            
            # Feature engineering
            df_processed = self.create_study_efficiency(df_processed)
            df_processed = self.create_lifestyle_score(df_processed)
            
            # Encode categorical variables
            df_processed = self.encode_categorical_variables(df_processed)
            
            # Scale numerical features
            numerical_columns.extend(['study_efficiency', 'lifestyle_score'])
            df_processed = self.scale_numerical_features(df_processed, numerical_columns)
            
            # Store feature names
            if is_training:
                self.feature_names = df_processed.columns.tolist()
                self.feature_names.remove('student_id')
                self.feature_names.remove('exam_score')
            
            # Split features and target
            X = df_processed.drop(['student_id', 'exam_score'], axis=1)
            y = df_processed['exam_score']
            
            logger.info("Data preprocessing completed successfully")
            return X, y
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
            
    def save_preprocessor(self, filepath: str) -> None:
        """Save the preprocessor object."""
        preprocessor_config = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        joblib.dump(preprocessor_config, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
        
    @staticmethod
    def load_preprocessor(filepath: str) -> 'DataPreprocessor':
        """Load a saved preprocessor object."""
        preprocessor = DataPreprocessor()
        config = joblib.load(filepath)
        preprocessor.scaler = config['scaler']
        preprocessor.label_encoders = config['label_encoders']
        preprocessor.feature_names = config['feature_names']
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor

if __name__ == "__main__":
    # Example usage
    try:
        # Load data
        df = pd.read_csv("data/student_habits_performance.csv")
        
        # Initialize and run preprocessor
        preprocessor = DataPreprocessor()
        X, y = preprocessor.preprocess_data(df)
        
        # Save preprocessor
        preprocessor.save_preprocessor("models/preprocessor.joblib")
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info("Sample of preprocessed features:")
        logger.info(X.head())
        
    except Exception as e:
        logger.error(f"Error in preprocessing script: {str(e)}")
        raise 