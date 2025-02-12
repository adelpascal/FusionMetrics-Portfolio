import pandas as pd
import numpy as np

class DataAnalyzer:
    def __init__(self, df):
        self.df = df
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns

    def get_summary_statistics(self):
        """Generate summary statistics for numerical columns."""
        if len(self.numeric_cols) == 0:
            return pd.DataFrame()  # Return empty DataFrame if no numerical columns

        try:
            summary = self.df[self.numeric_cols].describe()
            return summary
        except Exception as e:
            raise Exception(f"Error calculating summary statistics: {str(e)}")

    def get_missing_values(self):
        """Analyze missing values in the dataset."""
        try:
            missing = pd.DataFrame({
                'Column': self.df.columns,
                'Missing Values': self.df.isnull().sum(),
                'Percentage': (self.df.isnull().sum() / len(self.df) * 100).round(2)
            })
            return missing.sort_values('Missing Values', ascending=False)
        except Exception as e:
            raise Exception(f"Error analyzing missing values: {str(e)}")

    def get_correlations(self):
        """Calculate correlation matrix for numerical columns."""
        if len(self.numeric_cols) == 0:
            return pd.DataFrame()  # Return empty DataFrame if no numerical columns

        try:
            correlations = self.df[self.numeric_cols].corr()
            return correlations
        except Exception as e:
            raise Exception(f"Error calculating correlations: {str(e)}")

    def get_column_types(self):
        """Get data types of all columns."""
        try:
            return pd.DataFrame({
                'Column': self.df.columns,
                'Type': self.df.dtypes,
                'Unique Values': self.df.nunique()
            })
        except Exception as e:
            raise Exception(f"Error getting column types: {str(e)}")

    def get_categorical_summary(self):
        """Generate summary for categorical columns."""
        if len(self.categorical_cols) == 0:
            return pd.DataFrame()  # Return empty DataFrame if no categorical columns

        try:
            summaries = {}
            for col in self.categorical_cols:
                value_counts = self.df[col].value_counts().head(5)  # Top 5 most common values
                summaries[col] = {
                    'unique_values': self.df[col].nunique(),
                    'top_values': value_counts.to_dict()
                }
            return pd.DataFrame.from_dict(summaries, orient='index')
        except Exception as e:
            raise Exception(f"Error summarizing categorical data: {str(e)}")