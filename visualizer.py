
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import pandas as pd

class Visualizer:
    def __init__(self, df):
        self.df = df
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns

    def plot_correlation_matrix(self, corr_matrix):
        """Create an enhanced heatmap of correlation matrix."""
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            hoverongaps=False,
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Correlation Matrix Heatmap',
            height=700,
            width=900,
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            template='plotly_white'
        )
        return fig

    def plot_histogram(self, column=None):
        """Create enhanced histograms with KDE."""
        if not column:
            column = self.numeric_cols[0] if len(self.numeric_cols) > 0 else None
            
        if column:
            fig = ff.create_distplot(
                [self.df[column].dropna()], 
                [column],
                bin_size=(self.df[column].max() - self.df[column].min()) / 30,
                show_rug=False
            )
            fig.update_layout(
                title=f'Distribution of {column}',
                template='plotly_white',
                height=400
            )
            return fig
        return None

    def plot_boxplot(self, numeric_col=None, group_by=None):
        """Create enhanced box plots with optional grouping."""
        if not numeric_col:
            numeric_col = self.numeric_cols[0] if len(self.numeric_cols) > 0 else None

        if numeric_col:
            if group_by and group_by in self.categorical_cols:
                fig = px.box(
                    self.df,
                    x=group_by,
                    y=numeric_col,
                    color=group_by,
                    notched=True,
                    points='outliers'
                )
            else:
                fig = px.box(
                    self.df,
                    y=numeric_col,
                    notched=True,
                    points='outliers'
                )
            
            fig.update_layout(
                title=f'Box Plot of {numeric_col}',
                template='plotly_white',
                height=400
            )
            return fig
        return None

    def plot_scatter_matrix(self):
        """Create an enhanced scatter matrix."""
        if len(self.numeric_cols) >= 2:
            fig = px.scatter_matrix(
                self.df,
                dimensions=self.numeric_cols[:4],
                color=self.categorical_cols[0] if len(self.categorical_cols) > 0 else None,
                opacity=0.7
            )
            
            fig.update_layout(
                title='Scatter Matrix',
                height=800,
                width=800,
                template='plotly_white'
            )
            return fig
        return None

    def plot_time_series(self, date_col, value_col):
        """Create time series plot if date column exists."""
        if date_col in self.df.columns and value_col in self.numeric_cols:
            fig = px.line(
                self.df.sort_values(date_col),
                x=date_col,
                y=value_col,
                title=f'Time Series: {value_col} over {date_col}'
            )
            
            fig.update_layout(
                template='plotly_white',
                height=400,
                xaxis_title=date_col,
                yaxis_title=value_col
            )
            return fig
        return None

    def plot_pie_chart(self, column):
        """Create pie chart for categorical data."""
        if column in self.categorical_cols:
            value_counts = self.df[column].value_counts()
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f'Distribution of {column}'
            )
            
            fig.update_layout(
                template='plotly_white',
                height=500
            )
            return fig
        return None
