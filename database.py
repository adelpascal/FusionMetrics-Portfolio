import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from sqlalchemy.engine.url import make_url
import time
import numpy as np

# Get database URL from environment variable
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

# Parse the URL and add SSL mode
url = make_url(DATABASE_URL)
url = url.set(drivername="postgresql+psycopg2")
engine = create_engine(
    url,
    connect_args={
        "sslmode": "require",
        "connect_timeout": 30
    },
    pool_pre_ping=True,
    pool_recycle=3600
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    row_count = Column(Integer)
    column_count = Column(Integer)
    data_types = Column(JSON)
    summary_stats = Column(JSON)
    missing_values = Column(JSON)
    correlation_matrix = Column(JSON)
    ai_insights = Column(String)

MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

def init_db():
    """Initialize database with retries"""
    for attempt in range(MAX_RETRIES):
        try:
            Base.metadata.create_all(bind=engine)
            return
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise Exception(f"Failed to initialize database after {MAX_RETRIES} attempts: {str(e)}")
            time.sleep(RETRY_DELAY)

def process_dataframe(df):
    """Safely process DataFrame and extract statistics"""
    try:
        # Separate numerical and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Calculate summary statistics only for numerical columns
        summary_stats = {}
        if len(numeric_cols) > 0:
            summary_stats = df[numeric_cols].describe().to_dict()

        # Calculate correlations only for numerical columns
        correlation_matrix = {}
        if len(numeric_cols) > 0:
            correlation_matrix = df[numeric_cols].corr().to_dict()

        # Get missing values for all columns
        missing_values = df.isnull().sum().to_dict()

        return {
            'summary_stats': summary_stats,
            'correlation_matrix': correlation_matrix,
            'missing_values': missing_values
        }
    except Exception as e:
        raise Exception(f"Failed to process DataFrame: {str(e)}")

def save_analysis(filename: str, df: pd.DataFrame, ai_insights: str):
    """Save analysis results to database with retry logic"""
    for attempt in range(MAX_RETRIES):
        session = SessionLocal()
        try:
            # Process DataFrame safely
            processed_data = process_dataframe(df)

            # Create analysis object
            analysis = Analysis(
                filename=filename,
                row_count=len(df),
                column_count=len(df.columns),
                data_types={col: str(dtype) for col, dtype in df.dtypes.items()},
                summary_stats=processed_data['summary_stats'],
                missing_values=processed_data['missing_values'],
                correlation_matrix=processed_data['correlation_matrix'],
                ai_insights=ai_insights
            )

            # Add and commit in a single transaction
            session.add(analysis)
            session.flush()  # Ensure the object is flushed to get the ID
            analysis_id = analysis.id  # Get the ID before committing
            session.commit()
            session.close()
            return analysis_id

        except Exception as e:
            session.rollback()
            if attempt == MAX_RETRIES - 1:
                session.close()
                raise Exception(f"Failed to save analysis after {MAX_RETRIES} attempts: {str(e)}")
            time.sleep(RETRY_DELAY)
        finally:
            # Ensure session is closed even if an error occurs
            if session:
                session.close()

def get_analysis_history():
    """Retrieve analysis history with retry logic"""
    for attempt in range(MAX_RETRIES):
        session = SessionLocal()
        try:
            # Query all analyses
            analyses = session.query(Analysis).order_by(Analysis.timestamp.desc()).all()

            # Create result list while session is still open
            results = [
                {
                    "id": analysis.id,
                    "filename": analysis.filename,
                    "timestamp": analysis.timestamp,
                    "row_count": analysis.row_count,
                    "column_count": analysis.column_count
                }
                for analysis in analyses
            ]

            session.close()
            return results

        except Exception as e:
            session.rollback()
            if attempt == MAX_RETRIES - 1:
                session.close()
                raise Exception(f"Failed to retrieve analysis history after {MAX_RETRIES} attempts: {str(e)}")
            time.sleep(RETRY_DELAY)
        finally:
            # Ensure session is closed even if an error occurs
            if session:
                session.close()

    return []