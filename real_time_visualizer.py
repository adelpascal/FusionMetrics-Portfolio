import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time
from collections import deque
import threading
import queue

class RealTimeVisualizer:
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.data_queue = queue.Queue()
        self.time_series = deque(maxlen=max_points)
        self.values = deque(maxlen=max_points)
        self.stop_signal = threading.Event()
        
    def generate_sample_data(self):
        """Generate sample time series data."""
        while not self.stop_signal.is_set():
            timestamp = datetime.now()
            value = np.random.normal(0, 1)  # Random value from normal distribution
            self.data_queue.put((timestamp, value))
            time.sleep(0.1)  # Simulate data generation every 100ms
            
    def start_streaming(self):
        """Start the data streaming thread."""
        self.stop_signal.clear()
        self.stream_thread = threading.Thread(target=self.generate_sample_data)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        
    def stop_streaming(self):
        """Stop the data streaming."""
        self.stop_signal.set()
        if hasattr(self, 'stream_thread'):
            self.stream_thread.join()
            
    def update_plot_data(self):
        """Update plot data with new values from queue."""
        while not self.data_queue.empty():
            timestamp, value = self.data_queue.get()
            self.time_series.append(timestamp)
            self.values.append(value)
            
    def create_plot(self):
        """Create and return the real-time plot."""
        self.update_plot_data()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(self.time_series),
            y=list(self.values),
            mode='lines+markers',
            name='Real-time Data'
        ))
        
        fig.update_layout(
            title='Real-time Data Stream',
            xaxis_title='Time',
            yaxis_title='Value',
            height=400,
            showlegend=True,
            uirevision=True  # Preserve zoom level on updates
        )
        
        return fig
