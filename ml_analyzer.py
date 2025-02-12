import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import json

class MLAnalyzer:
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        self.target_column = None
        self.problem_type = None
        self.model = None
        self.label_encoders = {}
        self.feature_scaler = StandardScaler()
        
    def prepare_features(self, target_column):
        """Prepare features for ML model."""
        try:
            self.target_column = target_column
            X = self.df.drop(columns=[target_column])
            y = self.df[target_column]
            
            # Determine problem type
            if y.dtype == 'object' or len(np.unique(y)) < 10:
                self.problem_type = 'classification'
                # Encode target for classification
                le = LabelEncoder()
                y = le.fit_transform(y)
                self.label_encoders['target'] = le
            else:
                self.problem_type = 'regression'
            
            # Process features
            X_processed = pd.DataFrame()
            
            # Handle numeric features
            numeric_features = X.select_dtypes(include=[np.number])
            if not numeric_features.empty:
                X_processed = pd.DataFrame(
                    self.feature_scaler.fit_transform(numeric_features),
                    columns=numeric_features.columns
                )
            
            # Handle categorical features
            for col in X.select_dtypes(exclude=[np.number]):
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            
            return X_processed, y
            
        except Exception as e:
            raise Exception(f"Error preparing features: {str(e)}")
    
    def train_model(self, target_column, test_size=0.2):
        """Train ML model and return performance metrics."""
        try:
            # Prepare data
            X, y = self.prepare_features(target_column)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Select and train model
            if self.problem_type == 'classification':
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
                metric_name = 'accuracy'
            else:
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
                metric_name = 'r2_score'
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Get predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            metrics = {}
            if self.problem_type == 'classification':
                metrics['accuracy'] = accuracy_score(y_test, y_pred)
                metrics['n_classes'] = len(np.unique(y))
            else:
                metrics['r2_score'] = r2_score(y_test, y_pred)
                metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return {
                'problem_type': self.problem_type,
                'metric_name': metric_name,
                'metrics': metrics,
                'feature_importance': feature_importance.to_dict('records')
            }
            
        except Exception as e:
            raise Exception(f"Error training model: {str(e)}")
    
    def get_model_summary(self):
        """Get a summary of the trained model."""
        if not self.model:
            return "No model has been trained yet."
        
        summary = {
            'problem_type': self.problem_type,
            'target_column': self.target_column,
            'n_features': len(self.numeric_cols) + len(self.categorical_cols),
            'model_type': self.model.__class__.__name__
        }
        return summary
