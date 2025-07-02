# Simplified PHM Model Training Resource Analyzer

"""
PHM Training Resource Analyzer - Simplified model training resource analysis tool

Features:
- Automatic hardware environment detection (CPU/GPU/Memory)
- Input training data dimensions and target dimensions
- Analyze model structure parameters
- Estimate training time and resource consumption per epoch for specified batch_size

Core calculation logic:
- Training time estimation: Based on model parameters, data dimensions and hardware performance
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
    
    def calculate_model_params(self, input_dim, hidden_dims, output_dim, model_type="Dense"):
        """Calculate model parameters (millions) based on network structure"""
        if model_type == "Dense":
            # Fully connected network: input -> hidden1 -> hidden2 -> output
            if len(hidden_dims) == 0:
                params = input_dim * output_dim + output_dim  # weights + bias
            else:
                params = input_dim * hidden_dims[0] + hidden_dims[0]  # first layer
                for i in range(1, len(hidden_dims)):
                    params += hidden_dims[i-1] * hidden_dims[i] + hidden_dims[i]  # hidden layers
                params += hidden_dims[-1] * output_dim + output_dim  # output layer
        
        elif model_type == "CNN":
            # Simplified CNN parameter estimation
            conv_params = input_dim * 32 * 3 * 3 + 32 * 64 * 3 * 3  # two convolutional layers
            if len(hidden_dims) > 0:
                fc_params = 64 * hidden_dims[0] + hidden_dims[0]
                for i in range(1, len(hidden_dims)):
                    fc_params += hidden_dims[i-1] * hidden_dims[i] + hidden_dims[i]
                fc_params += hidden_dims[-1] * output_dim + output_dim
            else:
                fc_params = 64 * output_dim + output_dim
            params = conv_params + fc_params
        
        elif model_type == "LSTM":
            # LSTM parameter estimation: 4 * (input_size + hidden_size + 1) * hidden_size
            if len(hidden_dims) > 0:
                lstm_hidden = hidden_dims[0]
            else:
                lstm_hidden = 128  # default hidden layer size
            
            lstm_params = 4 * (input_dim + lstm_hidden + 1) * lstm_hidden
            fc_params = lstm_hidden * output_dim + output_dim
            params = lstm_params + fc_params
        
        else:
            # Default estimation
            total_hidden = sum(hidden_dims) if hidden_dims else 256
            params = input_dim * total_hidden + total_hidden * output_dim
        
        return params / 1e6  # Convert to millions of parameters
    
    def estimate_training_time(self, model_params_m, batch_size, input_dim):
        """Estimate training time per epoch (minutes)"""
        # Base time factor (GPU vs CPU)
        if self.gpu_available:
            base_time_per_million_params = 0.5  # GPU: 0.5 minutes/million parameters
            device_factor = 1.0
        else:
            base_time_per_million_params = 3.0  # CPU: 3 minutes/million parameters
            device_factor = 2.0
        
        # Batch size impact (smaller batches require more iterations)
        batch_factor = 32 / batch_size  # baseline batch_size=32
        
        # Input dimension impact
        dim_factor = max(1.0, input_dim / 1000)  # baseline 1000 dimensions
        
        # Calculate time per epoch
        time_per_epoch = (base_time_per_million_params * model_params_m * 
                         batch_factor * dim_factor * device_factor)
        
        return max(0.1, time_per_epoch)  # minimum 0.1 minutes
    
    def estimate_memory_usage(self, model_params_m, batch_size, input_dim, output_dim):
        """Estimate memory usage (GB)"""
        # Model parameter memory (float32: 4 bytes/parameter)
        model_memory = model_params_m * 4 / 1000  # MB to GB
        
        # Gradient memory (same as model parameters)
        gradient_memory = model_memory
        
        # Data batch memory (input + output)
        data_memory = batch_size * (input_dim + output_dim) * 4 / (1024**3)
        
        # Optimizer state memory (Adam: approximately 2x model parameters)
        optimizer_memory = model_memory * 2
        
        # Other overhead (cache, intermediate results, etc.)
        overhead_memory = model_memory * 0.5
        
        total_memory = (model_memory + gradient_memory + data_memory + 
                       optimizer_memory + overhead_memory)
        
        return {
            "model_memory": model_memory,
            "gradient_memory": gradient_memory,
            "data_memory": data_memory,
            "optimizer_memory": optimizer_memory,
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

# Create simplified GUI interface
def create_simple_gui():
    analyzer = SimplePHMAnalyzer()
    
    # Display hardware information
    hw_info = analyzer.get_hardware_info()
    print("Detected Hardware Environment:")
    print(f"  • CPU: {hw_info['cpu_cores']} cores ({hw_info['logical_cores']} logical cores)")
    print(f"  • RAM: {hw_info['ram_gb']} GB")
    print(f"  • GPU: {hw_info['gpu_name']}")
    if hw_info['gpu_available']:
        print(f"  • VRAM: {hw_info['gpu_memory_gb']:.1f} GB")
    print(f"  • System: {hw_info['platform']}")
    print("-" * 50)
    
    # Create input controls
    input_dim_text = widgets.IntText(
        value=1000,
        description='Input Dim:',
        style={'description_width': '100px'}
    )
    
    output_dim_text = widgets.IntText(
        value=10,
        description='Output Dim:',
        style={'description_width': '100px'}
    )
    
    model_type_dropdown = widgets.Dropdown(
        options=['Dense', 'CNN', 'LSTM'],
        value='Dense',
        description='Model Type:',
        style={'description_width': '100px'}
    )
    
    hidden_dims_text = widgets.Text(
        value="512,256,128",
        description='Hidden Layers:',
        placeholder='e.g.: 512,256,128',
        style={'description_width': '100px'}
    )
    
    batch_size_slider = widgets.IntSlider(
        value=32,
        min=1,
        max=512,
        step=1,
        description='Batch Size:',
        style={'description_width': '100px'}
    )
    
    analyze_button = widgets.Button(
        description='Analyze Training Resources',
        button_style='info',
        layout=widgets.Layout(width='200px')
    )
    
    output_area = widgets.Output()
    
    def on_analyze_click(b):
        with output_area:
            clear_output()
            
            # Get input values
            input_dim = input_dim_text.value
            output_dim = output_dim_text.value
            model_type = model_type_dropdown.value
            batch_size = batch_size_slider.value
            
            # Parse hidden layer dimensions
            try:
                hidden_dims_str = hidden_dims_text.value.strip()
                if hidden_dims_str:
                    hidden_dims = [int(x.strip()) for x in hidden_dims_str.split(',')]
                else:
                    hidden_dims = []
            except:
                hidden_dims = [256]
            
            # Calculate model parameters
            model_params_m = analyzer.calculate_model_params(
                input_dim, hidden_dims, output_dim, model_type
            )
            
            # Estimate training time
            time_per_epoch = analyzer.estimate_training_time(
                model_params_m, batch_size, input_dim
            )
            
            # Estimate memory usage
            memory_usage = analyzer.estimate_memory_usage(
                model_params_m, batch_size, input_dim, output_dim
            )
            
            # Calculate resource utilization
            resource_util = analyzer.calculate_resource_utilization(memory_usage)
            
            # Display results
            print("Training Resource Analysis Results")
            print("=" * 40)
            print(f"Model Configuration:")
            print(f"  • Type: {model_type}")
            print(f"  • Input Dimension: {input_dim}")
            print(f"  • Output Dimension: {output_dim}")
            print(f"  • Hidden Layers: {hidden_dims}")
            print(f"  • Total Parameters: {model_params_m:.2f}M")
            print(f"  • Batch Size: {batch_size}")
            print()
            
            print(f"Training Time Estimation:")
            print(f"  • Per Epoch: {time_per_epoch:.2f} minutes")
            print(f"  • 10 Epochs: {time_per_epoch*10:.1f} minutes ({time_per_epoch*10/60:.1f} hours)")
            print(f"  • 100 Epochs: {time_per_epoch*100:.1f} minutes ({time_per_epoch*100/60:.1f} hours)")
            print(f"  • Training Device: {resource_util['device_type']}")
            print()
            
            print(f"Memory Usage Estimation:")
            print(f"  • Model Parameters: {memory_usage['model_memory']:.3f} GB")
            print(f"  • Gradient Storage: {memory_usage['gradient_memory']:.3f} GB")
            print(f"  • Data Batch: {memory_usage['data_memory']:.3f} GB")
            print(f"  • Optimizer State: {memory_usage['optimizer_memory']:.3f} GB")
            print(f"  • Other Overhead: {memory_usage['overhead_memory']:.3f} GB")
            print(f"  • Total Required: {memory_usage['total_memory']:.3f} GB")
            print(f"  • Feasibility: {'Feasible' if memory_usage['feasible'] else 'Insufficient Memory'}")
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
                print("  Warning: Reduce batch size or model complexity")
                print("  Tip: Consider using gradient accumulation techniques")
            
            if resource_util['memory_utilization'] > 90:
                print("  Warning: Memory usage too high, recommend reducing")
            elif resource_util['memory_utilization'] < 50:
                print("  Tip: Memory utilization low, can increase batch size")
            
            if time_per_epoch > 60:
                print("  Note: Training time is long, consider using GPU acceleration")
            
            if not analyzer.gpu_available:
                print("  Recommendation: Use GPU for better training performance")
            
            print(f"\nCurrent Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    analyze_button.on_click(on_analyze_click)
    
    # Interface layout
    input_box = widgets.VBox([
        widgets.HTML("<h3>PHM Model Training Resource Analyzer</h3>"),
        input_dim_text,
        output_dim_text,
        model_type_dropdown,
        hidden_dims_text,
        batch_size_slider,
        analyze_button
    ])
    
    display(input_box)
    display(output_area)

# Start GUI
print("Simplified PHM Model Training Resource Analyzer")
print("=" * 50)
create_simple_gui()
