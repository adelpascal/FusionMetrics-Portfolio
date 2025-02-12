from fpdf import FPDF
import io

class PDFGenerator:
    def __init__(self):
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.add_page()

    def generate_report(self, df, summary_stats, missing_values, correlations, insights):
        """Generate a PDF report with analysis results."""
        # Set up the PDF
        self.pdf.set_font("Arial", "B", 16)
        self.pdf.cell(0, 10, "Data Analysis Report", ln=True, align="C")
        
        # Dataset Overview
        self.pdf.set_font("Arial", "B", 14)
        self.pdf.cell(0, 10, "Dataset Overview", ln=True)
        self.pdf.set_font("Arial", "", 12)
        self.pdf.cell(0, 10, f"Number of rows: {df.shape[0]}", ln=True)
        self.pdf.cell(0, 10, f"Number of columns: {df.shape[1]}", ln=True)
        
        # Summary Statistics
        self.pdf.add_page()
        self.pdf.set_font("Arial", "B", 14)
        self.pdf.cell(0, 10, "Summary Statistics", ln=True)
        self.pdf.set_font("Arial", "", 10)
        self._add_dataframe(summary_stats)
        
        # Missing Values
        self.pdf.add_page()
        self.pdf.set_font("Arial", "B", 14)
        self.pdf.cell(0, 10, "Missing Values Analysis", ln=True)
        self.pdf.set_font("Arial", "", 10)
        self._add_dataframe(missing_values)
        
        # AI Insights
        self.pdf.add_page()
        self.pdf.set_font("Arial", "B", 14)
        self.pdf.cell(0, 10, "AI-Generated Insights", ln=True)
        self.pdf.set_font("Arial", "", 12)
        self.pdf.multi_cell(0, 10, insights)
        
        # Generate PDF
        pdf_output = io.BytesIO()
        self.pdf.output(pdf_output)
        return pdf_output.getvalue()

    def _add_dataframe(self, df):
        """Helper method to add a dataframe to the PDF."""
        # Convert dataframe to string representation
        data_str = df.to_string()
        self.pdf.set_font("Courier", "", 8)
        self.pdf.multi_cell(0, 5, data_str)
