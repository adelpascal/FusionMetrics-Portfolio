import os
from openai import OpenAI
import json

class AIInsights:
    def __init__(self, df):
        self.df = df
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-4o"
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def generate_insights(self):
        """Generate AI-powered insights from the dataset."""
        # Prepare dataset summary
        data_summary = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "summary_stats": self.df.describe().to_dict(),
            "missing_values": self.df.isnull().sum().to_dict()
        }

        prompt = f"""Analyze this dataset and provide key insights in markdown format:
        Dataset Summary: {json.dumps(data_summary)}
        
        Please provide insights in the following areas:
        1. Data Quality
        2. Key Statistics
        3. Patterns and Relationships
        4. Recommendations
        
        Format the response in markdown with clear sections and bullet points."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data analysis expert providing clear, actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating insights: {str(e)}"
