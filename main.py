import streamlit as st
import pandas as pd
from data_analyzer import DataAnalyzer
from visualizer import Visualizer
from ai_insights import AIInsights
from pdf_generator import PDFGenerator
from utils import validate_file, show_progress
from database import save_analysis, get_analysis_history, init_db
import os
from ml_analyzer import MLAnalyzer
from real_time_visualizer import RealTimeVisualizer
import time

# Initialize database
init_db()

st.set_page_config(
    page_title="Enterprise Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise styling
st.markdown("""
    <style>
    .main-header {
        background-color: #f8f9fa;
        padding: 1.5rem 1rem;
        border-bottom: 1px solid #eaecef;
        margin-bottom: 2rem;
    }
    .sub-header {
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    .enterprise-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Enterprise header
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🔍 Enterprise Data Analyzer")
st.markdown('<p class="sub-header">Advanced Analytics & Machine Learning Platform</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        border: none;
        font-weight: bold;
    }
    .stSelectbox {
        min-width: 200px;
    }
    .uploadedFile {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #fff;
        border-radius: 5px;
        color: #000;
        font-size: 14px;
        font-weight: 400;
        align-items: center;
        justify-content: center;
        border: 1px solid #ddd;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B !important;
        color: white !important;
        border: 1px solid #FF4B4B !important;
    }
    </style>
""", unsafe_allow_html=True)

def check_openai_api_key():
    """Check if OpenAI API key is available and valid."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("""
        OpenAI API key is missing. To enable AI-powered insights:
        1. Get your API key from https://platform.openai.com/account/api-keys
        2. Add it to the project's secrets
        """)
        return False
    return True

def main():
    st.title("📊 Instant Data Analyst")
    st.markdown("""
    Upload your dataset (CSV or Excel) and get instant AI-powered insights and visualizations.
    """)

    # Create tabs for upload and history with icons
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 📤 Upload your dataset for instant analysis")
    with col2:
        st.markdown("### 📚 View History")
    
    tab_upload, tab_history = st.tabs(["📤 Upload Dataset", "📚 Analysis History"])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx'],
            help="Upload a CSV or Excel file"
        )

        if uploaded_file is not None:
            try:
                # Validate file
                if not validate_file(uploaded_file):
                    st.error("Invalid file format or size. Please upload a valid CSV or Excel file under 100MB.")
                    return

                with st.spinner("Reading data..."):
                    # Read the file based on its type
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)

                # Initialize components
                analyzer = DataAnalyzer(df)
                visualizer = Visualizer(df)
                pdf_gen = PDFGenerator()

                # Create tabs for different sections
                analysis_tabs = st.tabs(["Summary", "Visualizations", "AI Insights", "ML Analysis", "Real-time", "Export"])

                with analysis_tabs[0]:
                    st.subheader("📑 Data Summary")
                    with show_progress("Analyzing data..."):
                        # Get data summaries
                        summary_stats = analyzer.get_summary_statistics()
                        missing_values = analyzer.get_missing_values()
                        correlations = analyzer.get_correlations()
                        categorical_summary = analyzer.get_categorical_summary()
                        column_types = analyzer.get_column_types()

                    # Display basic dataset info
                    st.write("Dataset Overview")
                    st.write(f"Total Rows: {len(df)}")
                    st.write(f"Total Columns: {len(df.columns)}")

                    # Display column types
                    st.write("Column Types")
                    st.dataframe(column_types)

                    # Display numerical summaries if available
                    if not summary_stats.empty:
                        st.write("Numerical Data Statistics")
                        st.dataframe(summary_stats)

                    # Display categorical summaries if available
                    if not categorical_summary.empty:
                        st.write("Categorical Data Summary")
                        st.dataframe(categorical_summary)

                    st.write("Missing Values Analysis")
                    st.dataframe(missing_values)

                    # Show correlation matrix only if numerical columns exist
                    if not correlations.empty:
                        st.write("Correlation Matrix")
                        st.plotly_chart(visualizer.plot_correlation_matrix(correlations))

                with analysis_tabs[1]:
                    st.subheader("📈 Visualizations")
                    with show_progress("Generating visualizations..."):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(visualizer.plot_histogram())
                        with col2:
                            st.plotly_chart(visualizer.plot_boxplot())

                        st.plotly_chart(visualizer.plot_scatter_matrix())

                with analysis_tabs[2]:
                    st.subheader("🤖 AI Insights")
                    insights = None
                    if check_openai_api_key():
                        with show_progress("Generating AI insights..."):
                            try:
                                ai_insights = AIInsights(df)
                                insights = ai_insights.generate_insights()
                                st.markdown(insights)
                            except Exception as e:
                                st.error(f"Failed to generate AI insights: {str(e)}")
                                insights = "Error generating insights"
                    else:
                        insights = "OpenAI API key not configured"

                with analysis_tabs[3]:
                    st.subheader("🤖 Machine Learning Analysis")
                    if len(df) > 0:
                        ml_analyzer = MLAnalyzer(df)

                        # Select target column
                        target_column = st.selectbox(
                            "Select Target Column for Prediction",
                            df.columns.tolist(),
                            help="Choose the column you want to predict"
                        )

                        if st.button("Train Model"):
                            with st.spinner("Training model..."):
                                try:
                                    results = ml_analyzer.train_model(target_column)

                                    # Display results
                                    st.success(f"Model trained successfully! Problem type: {results['problem_type']}")

                                    # Show metrics
                                    metrics = results['metrics']
                                    st.write("### Model Performance")
                                    for metric, value in metrics.items():
                                        st.metric(metric, f"{value:.4f}")

                                    # Show feature importance
                                    st.write("### Feature Importance")
                                    importance_df = pd.DataFrame(results['feature_importance'])
                                    st.bar_chart(importance_df.set_index('feature')['importance'])

                                    # Save model info to the database
                                    model_summary = ml_analyzer.get_model_summary()

                                except Exception as e:
                                    st.error(f"Error training model: {str(e)}")
                    else:
                        st.info("Please upload a dataset to perform ML analysis")


                with analysis_tabs[4]:
                    st.subheader("📊 Real-time Data Streaming")

                    # Initialize the real-time visualizer in session state if not present
                    if 'real_time_viz' not in st.session_state:
                        st.session_state.real_time_viz = RealTimeVisualizer()

                    # Control buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Start Streaming"):
                            st.session_state.real_time_viz.start_streaming()
                            st.session_state.streaming = True

                    with col2:
                        if st.button("Stop Streaming"):
                            st.session_state.real_time_viz.stop_streaming()
                            st.session_state.streaming = False

                    # Create placeholder for the plot
                    plot_placeholder = st.empty()

                    # Update plot if streaming is active
                    if st.session_state.get('streaming', False):
                        while True:
                            # Update plot
                            fig = st.session_state.real_time_viz.create_plot()
                            plot_placeholder.plotly_chart(fig, use_container_width=True)
                            time.sleep(0.1)  # Update every 100ms

                            # Check if streaming should continue
                            if not st.session_state.get('streaming', False):
                                break

                with analysis_tabs[5]:
                    st.subheader("📄 Export Report")
                    if st.button("Generate PDF Report"):
                        with show_progress("Generating PDF report..."):
                            pdf_content = pdf_gen.generate_report(
                                df,
                                summary_stats,
                                missing_values,
                                correlations,
                                insights or "No insights available"
                            )

                            st.download_button(
                                label="Download PDF Report",
                                data=pdf_content,
                                file_name="data_analysis_report.pdf",
                                mime="application/pdf"
                            )

                # Save analysis to database
                with show_progress("Saving analysis..."):
                    save_analysis(uploaded_file.name, df, insights or "No insights available")
                    st.success("Analysis saved successfully!")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    with tab_history:
        st.subheader("📚 Previous Analyses")
        history = get_analysis_history()

        if not history:
            st.info("No previous analyses found. Upload a dataset to get started!")
        else:
            for analysis in history:
                with st.expander(f"{analysis['filename']} - {analysis['timestamp']}"):
                    st.write(f"Rows: {analysis['row_count']}")
                    st.write(f"Columns: {analysis['column_count']}")
                    st.write(f"Analyzed on: {analysis['timestamp']}")

if __name__ == "__main__":
    main()