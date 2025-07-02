# Simplified PHM Model Training Resource Analyzer

"""
PHM Training Resource Analyzer - Simplified model training resource analysis tool

Features:
- Automatic hardware environment detection (CPU/GPU/Memory)
- Analyze existing TensorFlow/Keras model structure parameters
- Estimate training time and resource consumption per epoch for specified batch_size
- Support for complex architectures with multiple outputs

Core calculation logic:
- Training time estimation: Based on actual model parameters, data dimensions and hardware performance
- Memory usage estimation: Model parameters + gradients + data batches + optimizer state
- Hardware resource detection: CPU cores, memory capacity, GPU availability and VRAM
"""

import numpy as np
import psutil
import platform
import torch
import warnings
import ipywidgets as widgets
from IPython.display import display, clear_output
from datetime import datetime
import time

warnings.filterwarnings('ignore')

class SimplePHMAnalyzer:
    def __init__(self):
        self.gpu_available = False
        self.gpu_memory_gb = 0
        self.gpu_name = "No GPU"
        self.cpu_count = psutil.cpu_count(logical=False)
        self.cpu_count_logical = psutil.cpu_count(logical=True)
        self.ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        
        # Detect hardware
        self._detect_hardware()
    
    def _detect_hardware(self):
        """Detect available hardware resources"""
        try:
            if torch.cuda.is_available():
                self.gpu_available = True
                self.gpu_name = torch.cuda.get_device_name(0)
                self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            self.gpu_available = False
        
        # Also try to detect TensorFlow GPU
        try:
            import tensorflow as tf
            if tf.config.list_physical_devices('GPU'):
                self.gpu_available = True
                if self.gpu_name == "No GPU":
                    self.gpu_name = "TensorFlow GPU Available"
        except:
            pass
    
    def analyze_keras_model(self, model, X_train_shape, y_train_shapes, batch_size):
        """Analyze existing Keras model parameters and estimate resources"""
        try:
            # Get model parameters
            total_params = model.count_params()
            trainable_params = sum([np.prod(w.shape) for layer in model.layers for w in layer.get_weights()])
            
            model_params_m = total_params / 1e6
            
            # Extract data dimensions - support multiple input dimensions
            if isinstance(X_train_shape, (list, tuple)) and len(X_train_shape) > 0:
                if len(X_train_shape) == 2:  # (samples, features)
                    input_features = X_train_shape[1]
                    samples = X_train_shape[0]
                elif len(X_train_shape) == 3:  # (samples, timesteps, features)
                    input_features = X_train_shape[1] * X_train_shape[2]
                    samples = X_train_shape[0]
                else:
                    input_features = np.prod(X_train_shape[1:])
                    samples = X_train_shape[0]
            else:
                raise ValueError("Invalid X_train_shape format")
            
            # Handle multiple outputs
            total_output_features = 0
            if isinstance(y_train_shapes, (list, tuple)):
                if isinstance(y_train_shapes[0], (list, tuple)):
                    # Multiple outputs: [(samples, features1), (samples, features2), ...]
                    for y_shape in y_train_shapes:
                        if len(y_shape) == 1:  # (samples,) - single value output
                            total_output_features += 1
                        else:  # (samples, features) - multi-dimensional output
                            total_output_features += y_shape[1] if len(y_shape) > 1 else 1
                else:
                    # Single output: (samples, features) or (samples,)
                    if len(y_train_shapes) == 1:
                        total_output_features = 1
                    else:
                        total_output_features = y_train_shapes[1] if len(y_train_shapes) > 1 else 1
            else:
                total_output_features = 1
            
            return {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "model_params_m": model_params_m,
                "input_features": input_features,
                "output_features": total_output_features,
                "samples": samples,
                "batch_size": batch_size,
                "batches_per_epoch": samples // batch_size,
                "num_outputs": len(y_train_shapes) if isinstance(y_train_shapes[0], (list, tuple)) else 1
            }
        except Exception as e:
            print(f"Error analyzing model: {e}")
            return None
    
    def estimate_training_time_keras(self, model_info):
        """Estimate training time per epoch for Keras model (minutes)"""
        model_params_m = model_info["model_params_m"]
        batch_size = model_info["batch_size"]
        input_features = model_info["input_features"]
        batches_per_epoch = model_info["batches_per_epoch"]
        num_outputs = model_info["num_outputs"]
        
        # Base time factor (GPU vs CPU)
        if self.gpu_available:
            base_time_per_million_params = 0.3  # GPU: 0.3 minutes/million parameters
            device_factor = 1.0
        else:
            base_time_per_million_params = 2.5  # CPU: 2.5 minutes/million parameters
            device_factor = 3.0
        
        # Batch size impact (smaller batches require more iterations)
        batch_factor = max(0.5, 64 / batch_size)  # baseline batch_size=64
        
        # Input features impact
        feature_factor = max(1.0, input_features / 500)  # baseline 500 features
        
        # Number of batches impact
        batch_count_factor = batches_per_epoch / 100  # baseline 100 batches per epoch
        
        # Multiple outputs impact
        output_factor = max(1.0, num_outputs * 0.3 + 0.7)  # each additional output adds complexity
        
        # Calculate time per epoch
        time_per_epoch = (base_time_per_million_params * model_params_m * 
                         batch_factor * feature_factor * batch_count_factor * device_factor * output_factor)
        
        return max(0.05, time_per_epoch)  # minimum 0.05 minutes (3 seconds)
    
    def estimate_memory_usage_keras(self, model_info):
        """Estimate memory usage for Keras model (GB)"""
        model_params_m = model_info["model_params_m"]
        batch_size = model_info["batch_size"]
        input_features = model_info["input_features"]
        output_features = model_info["output_features"]
        num_outputs = model_info["num_outputs"]
        
        # Model parameter memory (float32: 4 bytes/parameter)
        model_memory = model_params_m * 4 / 1000  # MB to GB
        
        # Gradient memory (same as model parameters)
        gradient_memory = model_memory
        
        # Data batch memory (input + all outputs)
        data_memory = batch_size * (input_features + output_features) * 4 / (1024**3)
        
        # Optimizer state memory (Adam: approximately 2x model parameters)
        optimizer_memory = model_memory * 2
        
        # Intermediate activations (depends on model depth and batch size)
        activation_memory = batch_size * input_features * 4 / (1024**3) * 2  # estimated
        
        # Multiple output overhead
        multi_output_overhead = num_outputs * 0.1 if num_outputs > 1 else 0
        
        # Other overhead (TensorFlow/Keras overhead, cache, etc.)
        overhead_memory = model_memory * 0.8 + multi_output_overhead
        
        total_memory = (model_memory + gradient_memory + data_memory + 
                       optimizer_memory + activation_memory + overhead_memory)
        
        return {
            "model_memory": model_memory,
            "gradient_memory": gradient_memory,
            "data_memory": data_memory,
            "optimizer_memory": optimizer_memory,
            "activation_memory": activation_memory,
            "overhead_memory": overhead_memory,
            "total_memory": total_memory,
            "feasible": total_memory <= (self.gpu_memory_gb if self.gpu_available else self.ram_gb * 0.8)
        }
    
    def get_hardware_info(self):
        """Return hardware information"""
        return {
            "cpu_cores": self.cpu_count,
            "logical_cores": self.cpu_count_logical,
            "ram_gb": self.ram_gb,
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "gpu_memory_gb": self.gpu_memory_gb,
            "platform": platform.platform()
        }
    
    def calculate_resource_utilization(self, memory_usage):
        """Calculate resource utilization"""
        if self.gpu_available:
            memory_util = (memory_usage["total_memory"] / self.gpu_memory_gb) * 100
            device_type = "GPU"
            total_memory = self.gpu_memory_gb
        else:
            memory_util = (memory_usage["total_memory"] / (self.ram_gb * 0.8)) * 100
            device_type = "CPU"
            total_memory = self.ram_gb
        
        cpu_util = min(100, (self.cpu_count / 4) * 25)  # Estimated CPU usage
        
        return {
            "device_type": device_type,
            "memory_utilization": memory_util,
            "cpu_utilization": cpu_util,
            "total_available_memory": total_memory
        }
    
    def monitor_training_epoch(self, model_info, epoch_start_time=None):
        """Monitor actual training time and resource usage for one epoch"""
        if epoch_start_time is None:
            epoch_start_time = time.time()
        
        # Get current memory usage
        memory_info = psutil.virtual_memory()
        current_memory_gb = (memory_info.total - memory_info.available) / (1024**3)
        
        # Get current CPU usage
        current_cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            "epoch_start_time": epoch_start_time,
            "current_memory_gb": current_memory_gb,
            "current_cpu_percent": current_cpu_percent,
            "available_memory_gb": memory_info.available / (1024**3)
        }
    
    def analyze_existing_model(self, model, X_train_shape, y_train_shapes, batch_size):
        """Complete analysis for existing Keras model"""
        print("PHM Model Training Resource Analysis")
        print("=" * 50)
        
        # Display hardware information
        hw_info = self.get_hardware_info()
        print("Detected Hardware Environment:")
        print(f"  • CPU: {hw_info['cpu_cores']} cores ({hw_info['logical_cores']} logical cores)")
        print(f"  • RAM: {hw_info['ram_gb']} GB")
        print(f"  • GPU: {hw_info['gpu_name']}")
        if hw_info['gpu_available']:
            print(f"  • VRAM: {hw_info['gpu_memory_gb']:.1f} GB")
        print(f"  • System: {hw_info['platform']}")
        print("-" * 50)
        
        # Analyze model
        model_info = self.analyze_keras_model(model, X_train_shape, y_train_shapes, batch_size)
        if model_info is None:
            print("Failed to analyze model!")
            return
        
        # Estimate training time
        time_per_epoch = self.estimate_training_time_keras(model_info)
        
        # Estimate memory usage
        memory_usage = self.estimate_memory_usage_keras(model_info)
        
        # Calculate resource utilization
        resource_util = self.calculate_resource_utilization(memory_usage)
        
        # Display results
        print("Model Architecture Analysis:")
        print(f"  • Total Parameters: {model_info['total_params']:,} ({model_info['model_params_m']:.2f}M)")
        print(f"  • Trainable Parameters: {model_info['trainable_params']:,}")
        print(f"  • Input Shape: {X_train_shape}")
        print(f"  • Output Shapes: {y_train_shapes}")
        print(f"  • Number of Outputs: {model_info['num_outputs']}")
        print(f"  • Training Samples: {model_info['samples']:,}")
        print(f"  • Batch Size: {batch_size}")
        print(f"  • Batches per Epoch: {model_info['batches_per_epoch']}")
        print()
        
        print(f"Training Time Estimation:")
        print(f"  • Per Epoch: {time_per_epoch:.2f} minutes ({time_per_epoch*60:.1f} seconds)")
        print(f"  • 10 Epochs: {time_per_epoch*10:.1f} minutes ({time_per_epoch*10/60:.1f} hours)")
        print(f"  • 100 Epochs: {time_per_epoch*100:.1f} minutes ({time_per_epoch*100/60:.1f} hours)")
        print(f"  • Training Device: {resource_util['device_type']}")
        print()
        
        print(f"Memory Usage Estimation:")
        print(f"  • Model Parameters: {memory_usage['model_memory']:.3f} GB")
        print(f"  • Gradient Storage: {memory_usage['gradient_memory']:.3f} GB")
        print(f"  • Data Batch: {memory_usage['data_memory']:.3f} GB")
        print(f"  • Optimizer State: {memory_usage['optimizer_memory']:.3f} GB")
        print(f"  • Activations: {memory_usage['activation_memory']:.3f} GB")
        print(f"  • Framework Overhead: {memory_usage['overhead_memory']:.3f} GB")
        print(f"  • Total Required: {memory_usage['total_memory']:.3f} GB")
        print(f"  • Feasibility: {'✓ Feasible' if memory_usage['feasible'] else '✗ Insufficient Memory'}")
        print()
        
        print(f"Resource Utilization:")
        print(f"  • Device Type: {resource_util['device_type']}")
        print(f"  • Memory Usage: {resource_util['memory_utilization']:.1f}%")
        print(f"  • CPU Usage: {resource_util['cpu_utilization']:.1f}%")
        print(f"  • Total Available Memory: {resource_util['total_available_memory']:.1f} GB")
        print()
        
        # Optimization suggestions
        print("Optimization Suggestions:")
        if not memory_usage['feasible']:
            print("  ⚠️  Memory insufficient - reduce batch size or model complexity")
            print("  💡  Consider gradient accumulation or model parallelism")
        
        if resource_util['memory_utilization'] > 90:
            print("  ⚠️  Memory usage very high - consider reducing batch size")
        elif resource_util['memory_utilization'] < 50:
            print("  💡  Memory utilization low - can increase batch size for faster training")
        
        if time_per_epoch > 60:
            print("  💡  Long training time - consider GPU acceleration or model optimization")
        
        if not self.gpu_available:
            print("  🚀  Recommendation: Use GPU for significantly better performance")
        
        print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return {
            "model_info": model_info,
            "time_per_epoch": time_per_epoch,
            "memory_usage": memory_usage,
            "resource_util": resource_util
        }

# Create function to analyze user's existing model
def analyze_phm_model(model, X_train_shape, y_train_shapes, batch_size):
    """
    Analyze your existing PHM model for training resource requirements
    
    Parameters:
    - model: Your compiled Keras model (e.g., model_prognostic)
    - X_train_shape: Shape tuple of your training data, e.g., (9635, 21) for 2D or (9635, 32, 21) for 3D
    - y_train_shapes: Shape tuple(s) of your labels:
        * For single output: (9635,) or (9635, 10)
        * For multiple outputs: [(9635, 128), (9635, 21)] - list of tuples for each output
    - batch_size: Your chosen batch size for training
    
    Example usage:
    # Single output model
    analyze_phm_model(model_prognostic, (9635, 21), (9635,), 64)
    
    # Multiple output model
    analyze_phm_model(model_prognostic, (9635, 21), [(9635, 128), (9635, 21)], 64)
    """
    analyzer = SimplePHMAnalyzer()
    return analyzer.analyze_existing_model(model, X_train_shape, y_train_shapes, batch_size)

# Function to monitor training in real-time
def monitor_training_progress(analyzer, model_info, epoch_num, epoch_start_time):
    """Monitor and print training progress for each epoch"""
    monitoring_info = analyzer.monitor_training_epoch(model_info, epoch_start_time)
    epoch_time = time.time() - epoch_start_time
    
    print(f"Epoch {epoch_num} completed:")
    print(f"  • Time: {epoch_time:.2f} seconds ({epoch_time/60:.2f} minutes)")
    print(f"  • Memory Usage: {monitoring_info['current_memory_gb']:.2f} GB")
    print(f"  • CPU Usage: {monitoring_info['current_cpu_percent']:.1f}%")
    print(f"  • Available Memory: {monitoring_info['available_memory_gb']:.2f} GB")

# Example usage function
def example_usage():
    """
    Example of how to use the analyzer with your model:
    
    # Single output example
    X_train_shape = (9635, 21)      # Your actual training data shape
    y_train_shape = (9635,)         # Your actual label shape (single output)
    batch_size = 64                 # Your chosen batch size
    
    analysis_results = analyze_phm_model(
        model_prognostic,           # Your trained model
        X_train_shape, 
        y_train_shape, 
        batch_size
    )
    
    # Multiple output example (based on your model with 2 outputs)
    X_train_shape = (9635, 21)                    # Input shape
    y_train_shapes = [(9635, 128), (9635, 21)]    # Two outputs: encoded_out1 and encoded_out2
    batch_size = 64
    
    analysis_results = analyze_phm_model(
        model_prognostic,
        X_train_shape,
        y_train_shapes,
        batch_size
    )
    
    # During training, monitor each epoch:
    # for epoch in range(num_epochs):
    #     epoch_start = time.time()
    #     # ... your training code ...
    #     monitor_training_progress(analyzer, analysis_results['model_info'], epoch, epoch_start)
    """
    pass

# Start analysis
print("Simplified PHM Model Training Resource Analyzer")
print("=" * 50)
print("Ready to analyze your Keras model!")
print("\nUsage:")
print("  analyze_phm_model(your_model, X_train_shape, y_train_shapes, batch_size)")
print("\nExamples:")
print("  # Single output:")
print("  analyze_phm_model(model_prognostic, (9635, 21), (9635,), 64)")
print("  # Multiple outputs:")
print("  analyze_phm_model(model_prognostic, (9635, 21), [(9635, 128), (9635, 21)], 64)")
