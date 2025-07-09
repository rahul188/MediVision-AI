"""
Logging utilities for MedGemma POC
Provides comprehensive timing and performance logging
"""

import time
import functools
import psutil
import streamlit as st
from loguru import logger
from datetime import datetime
from typing import Any, Callable, Dict, Optional
import sys
import os

# Configure logger
def setup_logger():
    """Setup logger with appropriate configuration"""
    # Remove default handler
    logger.remove()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Add file handler with rotation
    logger.add(
        "logs/medgemma_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        compression="zip"
    )
    
    # Add console handler for development
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | {message}"
    )
    
    return logger

# Initialize logger
setup_logger()

class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str, log_memory: bool = True):
        self.operation_name = operation_name
        self.log_memory = log_memory
        self.start_time = None
        self.start_memory = None
        
    def __enter__(self):
        self.start_time = time.time()
        if self.log_memory:
            self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        logger.info(f"🚀 Starting: {self.operation_name}")
        if self.log_memory and self.start_memory:
            logger.info(f"📊 Initial memory usage: {self.start_memory:.1f} MB")
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        
        if self.log_memory:
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_diff = end_memory - self.start_memory if self.start_memory else 0
            
            logger.info(f"✅ Completed: {self.operation_name}")
            logger.info(f"⏱️  Duration: {duration:.2f} seconds")
            logger.info(f"📈 Memory change: {memory_diff:+.1f} MB (now {end_memory:.1f} MB)")
        else:
            logger.info(f"✅ Completed: {self.operation_name} in {duration:.2f} seconds")
        
        # Log to Streamlit as well
        if hasattr(st, 'session_state'):
            if 'performance_logs' not in st.session_state:
                st.session_state.performance_logs = []
            
            st.session_state.performance_logs.append({
                'timestamp': datetime.now(),
                'operation': self.operation_name,
                'duration': duration,
                'memory_change': memory_diff if self.log_memory and self.start_memory else None
            })

def time_operation(operation_name: str = None, log_memory: bool = True):
    """Decorator to time function execution"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with PerformanceTimer(op_name, log_memory):
                result = func(*args, **kwargs)
            
            return result
        return wrapper
    return decorator

class MedGemmaLogger:
    """Specialized logger for MedGemma operations"""
    
    @staticmethod
    def log_model_loading(model_name: str, device: str):
        """Log model loading start"""
        logger.info(f"🤖 Loading model: {model_name} on {device}")
        
    @staticmethod
    def log_model_loaded(model_name: str, load_time: float, model_size_mb: float = None):
        """Log successful model loading"""
        logger.success(f"✅ Model loaded: {model_name} in {load_time:.2f}s")
        if model_size_mb:
            logger.info(f"📦 Model size: {model_size_mb:.1f} MB")
    
    @staticmethod
    def log_image_analysis_start(image_size: tuple, analysis_type: str):
        """Log image analysis start"""
        logger.info(f"🖼️  Starting image analysis: {analysis_type}")
        logger.info(f"🔍 Image dimensions: {image_size[0]}x{image_size[1]}")
    
    @staticmethod
    def log_image_analysis_complete(analysis_type: str, duration: float, token_count: int = None):
        """Log image analysis completion"""
        logger.success(f"✅ Image analysis complete: {analysis_type} in {duration:.2f}s")
        if token_count:
            logger.info(f"📝 Generated tokens: {token_count}")
            logger.info(f"⚡ Tokens/second: {token_count/duration:.1f}")
    
    @staticmethod
    def log_text_qa_start(question_length: int):
        """Log text Q&A start"""
        logger.info(f"💭 Starting text Q&A (question length: {question_length} chars)")
    
    @staticmethod
    def log_text_qa_complete(duration: float, response_length: int, token_count: int = None):
        """Log text Q&A completion"""
        logger.success(f"✅ Text Q&A complete in {duration:.2f}s")
        logger.info(f"📝 Response length: {response_length} chars")
        if token_count:
            logger.info(f"🔤 Generated tokens: {token_count}")
            logger.info(f"⚡ Tokens/second: {token_count/duration:.1f}")
    
    @staticmethod
    def log_report_generation_start(report_type: str, input_length: int):
        """Log report generation start"""
        logger.info(f"📄 Starting report generation: {report_type}")
        logger.info(f"📊 Input length: {input_length} chars")
    
    @staticmethod
    def log_report_generation_complete(report_type: str, duration: float, report_length: int, token_count: int = None):
        """Log report generation completion"""
        logger.success(f"✅ Report generation complete: {report_type} in {duration:.2f}s")
        logger.info(f"📋 Report length: {report_length} chars")
        if token_count:
            logger.info(f"🔤 Generated tokens: {token_count}")
            logger.info(f"⚡ Tokens/second: {token_count/duration:.1f}")
    
    @staticmethod
    def log_error(operation: str, error: Exception):
        """Log errors with context"""
        logger.error(f"❌ Error in {operation}: {str(error)}")
        logger.exception("Full error details:")
    
    @staticmethod
    def log_system_info():
        """Log system information"""
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        logger.info("🖥️  System Information:")
        logger.info(f"   💾 Total RAM: {memory.total / 1024**3:.1f} GB")
        logger.info(f"   🚀 Available RAM: {memory.available / 1024**3:.1f} GB")
        logger.info(f"   🔢 CPU Cores: {cpu_count}")
        
        # GPU info if available
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    logger.info(f"   🎮 GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                logger.info("   🎮 GPU: Not available")
        except ImportError:
            logger.info("   🎮 GPU: PyTorch not available")

def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary from session state"""
    if 'performance_logs' not in st.session_state:
        return {}
    
    logs = st.session_state.performance_logs
    if not logs:
        return {}
    
    # Calculate statistics
    durations = [log['duration'] for log in logs]
    operations = [log['operation'] for log in logs]
    
    summary = {
        'total_operations': len(logs),
        'total_time': sum(durations),
        'average_time': sum(durations) / len(durations),
        'fastest_operation': min(durations),
        'slowest_operation': max(durations),
        'operations_by_type': {}
    }
    
    # Group by operation type
    for log in logs:
        op_type = log['operation'].split(':')[0] if ':' in log['operation'] else log['operation']
        if op_type not in summary['operations_by_type']:
            summary['operations_by_type'][op_type] = {
                'count': 0,
                'total_time': 0,
                'durations': []
            }
        
        summary['operations_by_type'][op_type]['count'] += 1
        summary['operations_by_type'][op_type]['total_time'] += log['duration']
        summary['operations_by_type'][op_type]['durations'].append(log['duration'])
    
    # Calculate averages for each operation type
    for op_type in summary['operations_by_type']:
        op_data = summary['operations_by_type'][op_type]
        op_data['average_time'] = op_data['total_time'] / op_data['count']
    
    return summary

def display_performance_metrics():
    """Display performance metrics in Streamlit"""
    summary = get_performance_summary()
    
    if not summary:
        st.info("No performance data available yet. Start using the app to see metrics!")
        return
    
    st.subheader("📊 Performance Metrics")
    
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Operations", summary['total_operations'])
    
    with col2:
        st.metric("Total Time", f"{summary['total_time']:.1f}s")
    
    with col3:
        st.metric("Average Time", f"{summary['average_time']:.2f}s")
    
    with col4:
        fastest = summary['fastest_operation']
        slowest = summary['slowest_operation']
        st.metric("Time Range", f"{fastest:.1f}s - {slowest:.1f}s")
    
    # Operation breakdown
    if summary['operations_by_type']:
        st.subheader("🔍 Operation Breakdown")
        
        import pandas as pd
        
        breakdown_data = []
        for op_type, data in summary['operations_by_type'].items():
            breakdown_data.append({
                'Operation': op_type,
                'Count': data['count'],
                'Total Time (s)': round(data['total_time'], 2),
                'Average Time (s)': round(data['average_time'], 2),
                'Min Time (s)': round(min(data['durations']), 2),
                'Max Time (s)': round(max(data['durations']), 2)
            })
        
        df = pd.DataFrame(breakdown_data)
        st.dataframe(df, use_container_width=True)
    
    # Recent operations
    if 'performance_logs' in st.session_state and st.session_state.performance_logs:
        st.subheader("🕐 Recent Operations")
        
        recent_logs = st.session_state.performance_logs[-10:]  # Last 10 operations
        recent_data = []
        
        for log in reversed(recent_logs):  # Most recent first
            recent_data.append({
                'Time': log['timestamp'].strftime('%H:%M:%S'),
                'Operation': log['operation'],
                'Duration (s)': round(log['duration'], 2),
                'Memory Change (MB)': round(log['memory_change'], 1) if log['memory_change'] else 'N/A'
            })
        
        df_recent = pd.DataFrame(recent_data)
        st.dataframe(df_recent, use_container_width=True)

def clear_performance_logs():
    """Clear performance logs"""
    if 'performance_logs' in st.session_state:
        st.session_state.performance_logs = []
    logger.info("🧹 Performance logs cleared")

# Export logger instance
medgemma_logger = MedGemmaLogger()
