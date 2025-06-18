# 🚀 PHM Predictive Maintenance Training Environment Analysis & Resource Assessment

"""
PHM Training Analyzer - A comprehensive tool for analyzing and optimizing predictive maintenance model training configurations.

This analyzer provides intelligent estimation and optimization recommendations for training deep learning models 
in predictive maintenance scenarios. It evaluates hardware resources, estimates training times and memory usage, 
and provides task-specific recommendations for optimal performance.

Key Calculation Logic:
- **Training Time Estimation**: Uses baseline times for different model architectures (CNN-LSTM, Transformer, ResNet, VAE-LSTM)
  and adjusts based on dataset size, batch size, and model complexity using scaling factors
- **Memory Usage Calculation**: Estimates GPU/RAM requirements by calculating model parameters memory (4MB per million params),
  gradient memory (equal to model size), data memory (batch_size × input_size × 4 bytes), and optimizer memory (2x model size for Adam)
- **Hardware Detection**: Automatically detects CPU cores, RAM, GPU availability and memory to provide feasible configurations
- **Optimization Recommendations**: Provides task-specific batch size and epoch suggestions based on model architecture characteristics
  and dataset size patterns observed in predictive maintenance applications

Core Functions:
- Hardware resource detection and analysis
- Training time and memory estimation with safety margins
- Batch size and epoch optimization for different PHM tasks (bearing fault diagnosis, RUL prediction, health assessment, anomaly detection)
- Data dimension analysis with preprocessing overhead calculations
- Resource optimization strategies for CPU and GPU environments
- Interactive GUI for real-time configuration analysis and recommendations
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import platform
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import with fallback handling for optional dependencies
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
    print("Warning: PyTorch not available. GPU detection will be limited.")

try:
    import tensorflow as tf
    tf_available = True
except ImportError:
    tf_available = False
    print("Warning: TensorFlow not available. Some features may be limited.")

# Check if running in Jupyter/Colab environment
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        # Running in Jupyter/Colab
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
            widgets_available = True
        except ImportError:
            widgets_available = False
            print("Warning: ipywidgets not available. GUI functionality will be limited.")
    else:
        # Running as standalone script
        widgets_available = False
except ImportError:
    widgets_available = False

class PHMTrainingAnalyzer:
    def __init__(self):
        self.gpu_available = False
        self.gpu_memory_gb = 0
        self.gpu_name = "No GPU"
        self.cpu_count = psutil.cpu_count(logical=False)
        self.cpu_count_logical = psutil.cpu_count(logical=True)
        self.ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        
        # Detect hardware
        self._detect_hardware()
        
        # Pre-defined training configurations for reference
        self.base_configs = {
            "Bearing Fault Diagnosis": {
                "model_arch": "CNN-LSTM",
                "base_time_gpu": 3.0,
                "base_time_cpu": 20.0,
                "typical_input": "(2048, 1)",
                "typical_output": "4-8 classes",
                "preprocessing_factor": 1.5
            },
            "Remaining Useful Life (RUL)": {
                "model_arch": "Transformer",
                "base_time_gpu": 8.0,
                "base_time_cpu": 60.0,
                "typical_input": "(100, 14)",
                "typical_output": "1 continuous",
                "preprocessing_factor": 2.0
            },
            "Equipment Health Assessment": {
                "model_arch": "Deep ResNet",
                "base_time_gpu": 5.0,
                "base_time_cpu": 35.0,
                "typical_input": "(1024, 3)",
                "typical_output": "3-5 classes",
                "preprocessing_factor": 1.8
            },
            "Anomaly Detection": {
                "model_arch": "VAE-LSTM",
                "base_time_gpu": 2.0,
                "base_time_cpu": 15.0,
                "typical_input": "(512, 1)",
                "typical_output": "1 binary",
                "preprocessing_factor": 1.3
            }
        }
    
    def _detect_hardware(self):
        """Detect available hardware resources"""
        if torch_available and torch.cuda.is_available():
            self.gpu_available = True
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    def estimate_training_time(self, task_type, epochs, batch_size, dataset_size, model_params_m):
        """Estimate training time based on task configuration"""
        if task_type not in self.base_configs:
            return {"error": "Unknown task type"}
        
        config = self.base_configs[task_type]
        base_time = config["base_time_gpu"] if self.gpu_available else config["base_time_cpu"]
        
        # Adjust for batch size, dataset size, and model complexity
        size_factor = min(dataset_size / 10000, 3.0)  # Cap at 3x for very large datasets
        batch_factor = 32 / batch_size  # Normalize to batch size 32
        model_factor = model_params_m / 10.0  # Normalize to 10M parameters
        
        time_per_epoch = base_time * size_factor * batch_factor * model_factor
        total_time = time_per_epoch * epochs
        
        return {
            "time_per_epoch": f"{time_per_epoch:.1f} minutes",
            "total_time": f"{total_time:.1f} minutes ({total_time/60:.1f} hours)",
            "device_type": "GPU" if self.gpu_available else "CPU"
        }
    
    def estimate_memory_usage(self, batch_size, input_dims, output_dims, model_params_m):
        """Estimate memory usage based on user-provided parameters"""
        
        # Parse input dimensions (e.g., "(2048, 1)" -> 2048)
        try:
            input_size = eval(input_dims.replace("(", "").replace(")", "").split(",")[0])
        except:
            input_size = 1024  # Default fallback
        
        # Parse output dimensions (e.g., "4 classes" -> 4)
        try:
            if "classes" in output_dims.lower():
                output_size = int(output_dims.split()[0])
            elif "continuous" in output_dims.lower():
                output_size = 1
            else:
                output_size = int(output_dims)
        except:
            output_size = 1  # Default fallback
        
        # Memory estimation
        model_memory = model_params_m * 0.004  # ~4MB per M parameters
        gradient_memory = model_memory  # Gradients ≈ model size
        
        # Input data memory (batch_size * input_size * 4 bytes for float32)
        data_memory = (batch_size * input_size * 4) / (1024**3)  # Convert to GB
        
        # Additional overhead for optimizer states (Adam uses ~2x model params)
        optimizer_memory = model_memory * 2
        
        total_memory = model_memory + gradient_memory + data_memory + optimizer_memory
        
        return {
            "model_memory": f"{model_memory:.2f} GB",
            "data_memory": f"{data_memory:.3f} GB",
            "gradient_memory": f"{gradient_memory:.2f} GB",
            "optimizer_memory": f"{optimizer_memory:.2f} GB",
            "total_memory": f"{total_memory:.2f} GB",
            "feasible": total_memory <= self.ram_gb * 0.8  # Use 80% of available RAM
        }
    
    def get_hardware_info(self):
        """Return current hardware information"""
        return {
            "physical_cpu_cores": self.cpu_count,
            "logical_cpu_cores": self.cpu_count_logical,
            "total_ram_gb": self.ram_gb,
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "gpu_memory_gb": self.gpu_memory_gb,
            "platform": platform.platform()
        }
    
    def get_optimal_batch_size_suggestions(self, task_type, dataset_size, model_params_m, input_dims):
        """Provide optimal batch size suggestions based on hardware and task"""
        suggestions = []
        
        # Parse input dimensions for memory calculation
        try:
            input_size = eval(input_dims.replace("(", "").replace(")", "").split(",")[0])
        except:
            input_size = 1024
        
        # Calculate optimal batch sizes
        available_memory = self.gpu_memory_gb if self.gpu_available else self.ram_gb * 0.6
        
        # Estimate memory per sample
        model_memory = model_params_m * 0.004
        per_sample_memory = (input_size * 4) / (1024**3)  # Input data memory per sample
        
        # Calculate theoretical maximum batch size
        remaining_memory = available_memory - model_memory * 3  # Model + gradients + optimizer
        max_batch_size = int(remaining_memory / per_sample_memory)
        
        # Task-specific recommendations
        if task_type == "Bearing Fault Diagnosis":
            suggested_sizes = [16, 32, 64, 128]
            optimal_range = "32-64"
            reason = "CNN-LSTM works well with medium batch sizes for stable gradient updates"
        elif task_type == "Remaining Useful Life (RUL)":
            suggested_sizes = [8, 16, 32, 64]
            optimal_range = "16-32"
            reason = "Transformer models benefit from smaller batches for better generalization"
        elif task_type == "Equipment Health Assessment":
            suggested_sizes = [16, 32, 64, 128]
            optimal_range = "32-128"
            reason = "ResNet architectures can handle larger batches effectively"
        elif task_type == "Anomaly Detection":
            suggested_sizes = [32, 64, 128, 256]
            optimal_range = "64-128"
            reason = "VAE-LSTM benefits from larger batches for stable reconstruction loss"
        
        # Filter suggestions based on memory constraints
        feasible_sizes = [size for size in suggested_sizes if size <= max_batch_size]
        
        return {
            "suggested_sizes": feasible_sizes,
            "optimal_range": optimal_range,
            "max_theoretical": max_batch_size,
            "reason": reason,
            "memory_constraint": max_batch_size < min(suggested_sizes)
        }
    
    def get_optimal_epoch_suggestions(self, task_type, dataset_size, batch_size):
        """Provide optimal epoch suggestions based on task and data characteristics"""
        
        # Calculate steps per epoch
        steps_per_epoch = dataset_size // batch_size
        
        if task_type == "Bearing Fault Diagnosis":
            if dataset_size < 5000:
                epochs_range = "150-300"
                reason = "Small dataset needs more epochs for convergence"
            elif dataset_size < 20000:
                epochs_range = "100-200"
                reason = "Medium dataset with balanced training duration"
            else:
                epochs_range = "50-150"
                reason = "Large dataset converges faster"
        
        elif task_type == "Remaining Useful Life (RUL)":
            if dataset_size < 10000:
                epochs_range = "200-400"
                reason = "RUL prediction needs extensive training for pattern recognition"
            elif dataset_size < 50000:
                epochs_range = "150-300"
                reason = "Sufficient data for stable RUL model training"
            else:
                epochs_range = "100-200"
                reason = "Large dataset enables faster convergence"
        
        elif task_type == "Equipment Health Assessment":
            if dataset_size < 8000:
                epochs_range = "120-250"
                reason = "Health assessment needs balanced training"
            elif dataset_size < 30000:
                epochs_range = "80-180"
                reason = "Medium dataset with good representativeness"
            else:
                epochs_range = "60-120"
                reason = "Large dataset provides stable training"
        
        elif task_type == "Anomaly Detection":
            if dataset_size < 15000:
                epochs_range = "100-200"
                reason = "VAE models need sufficient training for reconstruction"
            elif dataset_size < 50000:
                epochs_range = "80-150"
                reason = "Balanced training for anomaly pattern learning"
            else:
                epochs_range = "50-100"
                reason = "Large dataset enables efficient anomaly detection training"
        
        return {
            "epochs_range": epochs_range,
            "steps_per_epoch": steps_per_epoch,
            "reason": reason,
            "early_stopping_patience": max(10, int(epochs_range.split("-")[1]) // 10)
        }
    
    def get_data_dimension_analysis(self, task_type, input_dims, output_dims, dataset_size):
        """Analyze data dimensions and provide optimization suggestions"""
        
        config = self.base_configs[task_type]
        typical_input = config["typical_input"]
        typical_output = config["typical_output"]
        preprocessing_factor = config["preprocessing_factor"]
        
        # Parse current dimensions
        try:
            current_input_size = eval(input_dims.replace("(", "").replace(")", "").split(",")[0])
        except:
            current_input_size = 1024
        
        # Parse typical dimensions for comparison
        try:
            typical_input_size = eval(typical_input.replace("(", "").replace(")", "").split(",")[0])
        except:
            typical_input_size = 1024
        
        suggestions = []
        
        # Input dimension analysis
        if current_input_size > typical_input_size * 2:
            suggestions.append(f"🔍 Consider dimensionality reduction: current {current_input_size} vs typical {typical_input_size}")
            suggestions.append("💡 Use PCA, feature selection, or downsampling to reduce input size")
        elif current_input_size < typical_input_size * 0.5:
            suggestions.append(f"📈 Input dimension might be too small: {current_input_size} vs typical {typical_input_size}")
            suggestions.append("💡 Consider feature engineering or data augmentation")
        
        # Preprocessing considerations
        preprocessing_memory = dataset_size * current_input_size * 4 * preprocessing_factor / (1024**3)
        if preprocessing_memory > self.ram_gb * 0.3:
            suggestions.append(f"⚠️ Preprocessing may require {preprocessing_memory:.1f}GB memory")
            suggestions.append("💡 Use data generators or batch preprocessing to manage memory")
        
        # Task-specific dimension suggestions
        if task_type == "Bearing Fault Diagnosis":
            if "1)" in input_dims:
                suggestions.append("🔧 Single-channel vibration data detected")
                suggestions.append("💡 Consider multi-channel data (acceleration, velocity, displacement)")
            suggestions.append("📊 Optimal window size: 1024-4096 samples for bearing analysis")
        
        elif task_type == "Remaining Useful Life (RUL)":
            if current_input_size < 50:
                suggestions.append("⏰ Time series might be too short for RUL prediction")
                suggestions.append("💡 Use longer time windows (100+ timesteps) for better trends")
            if "14)" in input_dims:
                suggestions.append("📈 Multi-sensor setup detected - good for RUL prediction")
        
        elif task_type == "Equipment Health Assessment":
            suggestions.append("🔍 Consider frequency domain features (FFT, wavelets)")
            suggestions.append("📊 Multi-resolution analysis can improve health assessment")
        
        elif task_type == "Anomaly Detection":
            suggestions.append("🎯 Ensure normal data is well-represented in training")
            suggestions.append("📉 Consider autoencoder-friendly preprocessing (normalization)")
        
        return {
            "suggestions": suggestions,
            "typical_input": typical_input,
            "typical_output": typical_output,
            "preprocessing_memory_gb": f"{preprocessing_memory:.2f}",
            "dimension_ratio": current_input_size / typical_input_size
        }
    
    def get_resource_optimization_strategy(self, task_type, dataset_size, model_params_m, memory_feasible):
        """Provide comprehensive resource optimization strategy"""
        
        strategy = []
        
        # Memory optimization
        if not memory_feasible:
            strategy.append("🔧 Memory Optimization Strategy:")
            strategy.append("  • Use gradient checkpointing to trade compute for memory")
            strategy.append("  • Implement mixed precision training (FP16)")
            strategy.append("  • Use data loaders with num_workers optimization")
            strategy.append("  • Consider model sharding for very large models")
        
        # CPU vs GPU optimization
        if self.gpu_available:
            strategy.append("🚀 GPU Optimization Strategy:")
            strategy.append("  • Use CuDNN optimized layers")
            strategy.append("  • Optimize data pipeline to avoid GPU starvation")
            strategy.append(f"  • Utilize all {self.gpu_memory_gb:.1f}GB GPU memory efficiently")
            strategy.append("  • Use asynchronous data loading")
        else:
            strategy.append("💻 CPU Optimization Strategy:")
            strategy.append(f"  • Utilize all {self.cpu_count} CPU cores")
            strategy.append("  • Use optimized BLAS libraries (MKL, OpenBLAS)")
            strategy.append("  • Consider smaller model architectures")
            strategy.append("  • Use model quantization for inference")
        
        # Dataset-specific strategies
        if dataset_size > 100000:
            strategy.append("📊 Large Dataset Strategies:")
            strategy.append("  • Use data sampling for initial experiments")
            strategy.append("  • Implement efficient data loading pipelines")
            strategy.append("  • Consider distributed training if available")
        
        # Task-specific optimizations
        config = self.base_configs[task_type]
        preprocessing_factor = config["preprocessing_factor"]
        
        strategy.append(f"🎯 {task_type} Specific Optimizations:")
        if task_type == "Bearing Fault Diagnosis":
            strategy.append("  • Use STFT preprocessing for frequency analysis")
            strategy.append("  • Implement class balancing for fault detection")
        elif task_type == "Remaining Useful Life (RUL)":
            strategy.append("  • Use sliding window approach for sequence data")
            strategy.append("  • Implement progressive training strategies")
        elif task_type == "Equipment Health Assessment":
            strategy.append("  • Use multi-scale feature extraction")
            strategy.append("  • Implement ensemble methods for robustness")
        elif task_type == "Anomaly Detection":
            strategy.append("  • Use reconstruction-based loss functions")
            strategy.append("  • Implement threshold optimization techniques")
        
        return strategy
    
    def get_recommendations(self, task_type, memory_feasible, training_time_hours):
        """Provide training recommendations based on analysis"""
        recommendations = []
        
        if not memory_feasible:
            recommendations.append("⚠️ Reduce batch size or use gradient checkpointing")
            recommendations.append("💡 Consider model pruning or quantization")
        
        if training_time_hours > 24:
            recommendations.append("⏰ Very long training time - consider reducing epochs")
            recommendations.append("🔄 Use learning rate scheduling and early stopping")
        
        if not self.gpu_available:
            recommendations.append("🚀 Enable GPU acceleration for faster training")
            recommendations.append("📊 Use mixed precision training if available")
        
        # Task-specific recommendations
        if "RUL" in task_type:
            recommendations.append("📈 Consider using transfer learning from pre-trained models")
            recommendations.append("🔍 Use cross-validation for robust evaluation")
        elif "Fault" in task_type:
            recommendations.append("🔧 Apply data augmentation for imbalanced classes")
            recommendations.append("📊 Use confusion matrix for detailed analysis")
        elif "Health" in task_type:
            recommendations.append("⚖️ Balance class distribution for better performance")
            recommendations.append("📈 Monitor validation metrics closely")
        elif "Anomaly" in task_type:
            recommendations.append("🎯 Use reconstruction loss for anomaly scoring")
            recommendations.append("📉 Consider threshold tuning for optimal performance")
        
        return recommendations

def analyze_configuration_console(task_type="Bearing Fault Diagnosis", 
                                input_dims="(2048, 1)", 
                                output_dims="4", 
                                model_params_m=10.0, 
                                epochs=100, 
                                batch_size=32, 
                                dataset_size=10000):
    """Console-based analysis function for command-line usage"""
    analyzer = PHMTrainingAnalyzer()
    
    # Display hardware info
    hw_info = analyzer.get_hardware_info()
    print("🖥️ Hardware Information:")
    print(f"  • CPU Cores: {hw_info['physical_cpu_cores']} physical, {hw_info['logical_cpu_cores']} logical")
    print(f"  • RAM: {hw_info['total_ram_gb']} GB")
    print(f"  • GPU: {hw_info['gpu_name']}")
    if hw_info['gpu_available']:
        print(f"  • GPU Memory: {hw_info['gpu_memory_gb']:.1f} GB")
    print(f"  • Platform: {hw_info['platform']}")
    print("-" * 60)
    
    # Perform analysis
    time_est = analyzer.estimate_training_time(task_type, epochs, batch_size, dataset_size, model_params_m)
    memory_est = analyzer.estimate_memory_usage(batch_size, input_dims, output_dims, model_params_m)
    
    # Get optimization suggestions
    batch_suggestions = analyzer.get_optimal_batch_size_suggestions(task_type, dataset_size, model_params_m, input_dims)
    epoch_suggestions = analyzer.get_optimal_epoch_suggestions(task_type, dataset_size, batch_size)
    dimension_analysis = analyzer.get_data_dimension_analysis(task_type, input_dims, output_dims, dataset_size)
    optimization_strategy = analyzer.get_resource_optimization_strategy(task_type, dataset_size, model_params_m, memory_est['feasible'])
    
    # Display results
    print("📊 Training Analysis Results:")
    print("=" * 50)
    print(f"Task: {task_type}")
    print(f"Model Architecture: {analyzer.base_configs[task_type]['model_arch']}")
    print(f"Input Dimensions: {input_dims}")
    print(f"Output Dimensions: {output_dims}")
    print(f"Model Parameters: {model_params_m}M")
    print(f"Configuration: {epochs} epochs, batch size {batch_size}")
    print(f"Dataset Size: {dataset_size:,} samples")
    print()
    
    print("⏱️ Time Estimation:")
    print(f"  • Per Epoch: {time_est['time_per_epoch']}")
    print(f"  • Total Training: {time_est['total_time']}")
    print(f"  • Device: {time_est['device_type']}")
    print()
    
    print("🧠 Memory Estimation:")
    print(f"  • Model Memory: {memory_est['model_memory']}")
    print(f"  • Data Memory: {memory_est['data_memory']}")
    print(f"  • Gradient Memory: {memory_est['gradient_memory']}")
    print(f"  • Optimizer Memory: {memory_est['optimizer_memory']}")
    print(f"  • Total Required: {memory_est['total_memory']}")
    print(f"  • Feasible: {'✅ Yes' if memory_est['feasible'] else '❌ No'}")
    print()
    
    # Optimization Suggestions
    print("🎯 Batch Size Optimization:")
    print(f"  • Recommended Range: {batch_suggestions['optimal_range']}")
    print(f"  • Feasible Sizes: {batch_suggestions['suggested_sizes']}")
    print(f"  • Max Theoretical: {batch_suggestions['max_theoretical']}")
    print(f"  • Rationale: {batch_suggestions['reason']}")
    if batch_suggestions['memory_constraint']:
        print("  ⚠️ Memory constraints detected - consider smaller batch sizes")
    print()
    
    print("📅 Epoch Optimization:")
    print(f"  • Recommended Range: {epoch_suggestions['epochs_range']}")
    print(f"  • Steps per Epoch: {epoch_suggestions['steps_per_epoch']}")
    print(f"  • Early Stopping Patience: {epoch_suggestions['early_stopping_patience']}")
    print(f"  • Rationale: {epoch_suggestions['reason']}")
    print()
    
    print("📏 Data Dimension Analysis:")
    print(f"  • Typical Input for Task: {dimension_analysis['typical_input']}")
    print(f"  • Typical Output for Task: {dimension_analysis['typical_output']}")
    print(f"  • Preprocessing Memory: {dimension_analysis['preprocessing_memory_gb']} GB")
    print(f"  • Dimension Ratio: {dimension_analysis['dimension_ratio']:.2f}x typical")
    print("  • Suggestions:")
    for suggestion in dimension_analysis['suggestions']:
        print(f"    {suggestion}")
    print()
    
    print("🔧 Resource Optimization Strategy:")
    for strategy_item in optimization_strategy:
        print(f"  {strategy_item}")
    print()
    
    # Get general recommendations
    training_hours = float(time_est['total_time'].split()[0]) / 60
    recommendations = analyzer.get_recommendations(
        task_type, memory_est['feasible'], training_hours
    )
    
    print("💡 Additional Recommendations:")
    for rec in recommendations:
        print(f"  {rec}")
    print()
    
    # Resource utilization
    memory_usage_percent = float(memory_est['total_memory'].split()[0]) / analyzer.ram_gb * 100
    print("📈 Resource Utilization Summary:")
    print(f"  • Memory Usage: {memory_usage_percent:.1f}% of available RAM")
    print(f"  • Estimated Peak Performance: {'GPU' if analyzer.gpu_available else 'CPU'} optimized")
    print(f"  • Preprocessing Overhead: {analyzer.base_configs[task_type]['preprocessing_factor']}x")
    
    # Final recommendations summary
    print("\n🎯 Final Configuration Recommendations:")
    optimal_batch = batch_suggestions['suggested_sizes'][len(batch_suggestions['suggested_sizes'])//2] if batch_suggestions['suggested_sizes'] else 32
    optimal_epochs = int(epoch_suggestions['epochs_range'].split('-')[0])
    print(f"  • Optimal Batch Size: {optimal_batch}")
    print(f"  • Optimal Epochs: {optimal_epochs}")
    print(f"  • Memory Safety Margin: {(analyzer.ram_gb * 0.8 - float(memory_est['total_memory'].split()[0])):.1f} GB remaining")
    
    # Training schedule suggestion
    current_hour = datetime.now().hour
    print(f"\n🕐 Current Time: {datetime.now().strftime('%H:%M')}")
    if 9 <= current_hour <= 17:
        print("  💼 Business hours - Good for debugging and hyperparameter tuning")
    else:
        print("  🌙 Off-hours - Ideal for long training runs")

# Create GUI Interface (only if widgets are available)
def create_phm_resource_assessment():
    if not widgets_available:
        print("⚠️ GUI not available. Use console version instead.")
        print("Example usage:")
        print("analyze_configuration_console()")
        return
    
    analyzer = PHMTrainingAnalyzer()
    
    # Display hardware info
    hw_info = analyzer.get_hardware_info()
    print("🖥️ Hardware Information:")
    print(f"  • CPU Cores: {hw_info['physical_cpu_cores']} physical, {hw_info['logical_cpu_cores']} logical")
    print(f"  • RAM: {hw_info['total_ram_gb']} GB")
    print(f"  • GPU: {hw_info['gpu_name']}")
    if hw_info['gpu_available']:
        print(f"  • GPU Memory: {hw_info['gpu_memory_gb']:.1f} GB")
    print(f"  • Platform: {hw_info['platform']}")
    print("-" * 60)
    
    # Create widgets
    task_dropdown = widgets.Dropdown(
        options=list(analyzer.base_configs.keys()),
        value=list(analyzer.base_configs.keys())[0],
        description='Task Type:',
        style={'description_width': 'initial'}
    )
    
    input_dims_text = widgets.Text(
        value="(2048, 1)",
        description='Input Dims:',
        placeholder='e.g., (2048, 1) or (100, 14)',
        style={'description_width': 'initial'}
    )
    
    output_dims_text = widgets.Text(
        value="4",
        description='Output Dims:',
        placeholder='e.g., 4 or 1 or 5 classes',
        style={'description_width': 'initial'}
    )
    
    model_params_text = widgets.FloatText(
        value=10.0,
        description='Model Params (M):',
        style={'description_width': 'initial'}
    )
    
    epochs_slider = widgets.IntSlider(
        value=100,
        min=10,
        max=500,
        step=10,
        description='Epochs:',
        style={'description_width': 'initial'}
    )
    
    batch_size_dropdown = widgets.Dropdown(
        options=[8, 16, 32, 64, 128, 256],
        value=32,
        description='Batch Size:',
        style={'description_width': 'initial'}
    )
    
    dataset_size_text = widgets.IntText(
        value=10000,
        description='Dataset Size:',
        style={'description_width': 'initial'}
    )
    
    analyze_button = widgets.Button(
        description='🔍 Analyze Training Configuration',
        button_style='info',
        layout=widgets.Layout(width='300px')
    )
    
    output_area = widgets.Output()
    
    def on_analyze_click(b):
        with output_area:
            clear_output()
            
            # Get input values
            task_type = task_dropdown.value
            input_dims = input_dims_text.value
            output_dims = output_dims_text.value
            model_params_m = model_params_text.value
            epochs = epochs_slider.value
            batch_size = batch_size_dropdown.value
            dataset_size = dataset_size_text.value
            
            # Use console analysis function
            analyze_configuration_console(task_type, input_dims, output_dims, 
                                        model_params_m, epochs, batch_size, dataset_size)
    
    analyze_button.on_click(on_analyze_click)
    
    # Layout
    input_box = widgets.VBox([
        widgets.HTML("<h3>🔧 PHM Training Configuration</h3>"),
        task_dropdown,
        widgets.HTML("<b>Model Specifications:</b>"),
        input_dims_text,
        output_dims_text,
        model_params_text,
        widgets.HTML("<b>Training Parameters:</b>"),
        epochs_slider,
        batch_size_dropdown,
        dataset_size_text,
        analyze_button
    ])
    
    display(input_box)
    display(output_area)

# Create simplified GUI interface for Colab
def create_phm_gui():
    """Simplified GUI interface that automatically loads widgets in Colab"""
    analyzer = PHMTrainingAnalyzer()
    
    # Display hardware info
    hw_info = analyzer.get_hardware_info()
    print("🖥️ Hardware Information:")
    print(f"  • CPU Cores: {hw_info['physical_cpu_cores']} physical, {hw_info['logical_cpu_cores']} logical")
    print(f"  • RAM: {hw_info['total_ram_gb']} GB")
    print(f"  • GPU: {hw_info['gpu_name']}")
    if hw_info['gpu_available']:
        print(f"  • GPU Memory: {hw_info['gpu_memory_gb']:.1f} GB")
    print(f"  • Platform: {hw_info['platform']}")
    print("-" * 60)
    
    # Import widgets and display here (assuming Colab environment)
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    
    # Create widgets
    task_dropdown = widgets.Dropdown(
        options=list(analyzer.base_configs.keys()),
        value=list(analyzer.base_configs.keys())[0],
        description='Task Type:',
        style={'description_width': 'initial'}
    )
    
    input_dims_text = widgets.Text(
        value="(2048, 1)",
        description='Input Dims:',
        placeholder='e.g., (2048, 1) or (100, 14)',
        style={'description_width': 'initial'}
    )
    
    output_dims_text = widgets.Text(
        value="4",
        description='Output Dims:',
        placeholder='e.g., 4 or 1 or 5 classes',
        style={'description_width': 'initial'}
    )
    
    model_params_text = widgets.FloatText(
        value=10.0,
        description='Model Params (M):',
        style={'description_width': 'initial'}
    )
    
    epochs_slider = widgets.IntSlider(
        value=100,
        min=10,
        max=500,
        step=10,
        description='Epochs:',
        style={'description_width': 'initial'}
    )
    
    batch_size_dropdown = widgets.Dropdown(
        options=[8, 16, 32, 64, 128, 256],
        value=32,
        description='Batch Size:',
        style={'description_width': 'initial'}
    )
    
    dataset_size_text = widgets.IntText(
        value=10000,
        description='Dataset Size:',
        style={'description_width': 'initial'}
    )
    
    analyze_button = widgets.Button(
        description='🔍 Analyze Training Configuration',
        button_style='info',
        layout=widgets.Layout(width='300px')
    )
    
    output_area = widgets.Output()
    
    def on_analyze_click(b):
        with output_area:
            clear_output()
            
            # Get input values
            task_type = task_dropdown.value
            input_dims = input_dims_text.value
            output_dims = output_dims_text.value
            model_params_m = model_params_text.value
            epochs = epochs_slider.value
            batch_size = batch_size_dropdown.value
            dataset_size = dataset_size_text.value
            
            # Use console analysis function
            analyze_configuration_console(task_type, input_dims, output_dims, 
                                        model_params_m, epochs, batch_size, dataset_size)
    
    analyze_button.on_click(on_analyze_click)
    
    # Layout
    input_box = widgets.VBox([
        widgets.HTML("<h3>🔧 PHM Training Configuration</h3>"),
        task_dropdown,
        widgets.HTML("<b>Model Specifications:</b>"),
        input_dims_text,
        output_dims_text,
        model_params_text,
        widgets.HTML("<b>Training Parameters:</b>"),
        epochs_slider,
        batch_size_dropdown,
        dataset_size_text,
        analyze_button
    ])
    
    display(input_box)
    display(output_area)

# Main execution
if __name__ == "__main__":
    print("🚀 PHM Predictive Maintenance Training Environment Analyzer")
    print("=" * 60)
    
    # Check environment and run appropriate interface
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            # Running in Jupyter/Colab - try to create GUI
            try:
                import ipywidgets as widgets
                from IPython.display import display, clear_output
                create_phm_gui()
                print("✅ Interactive GUI loaded successfully!")
                print("Use the widgets above to analyze different training configurations.")
            except ImportError:
                print("⚠️ ipywidgets not available. Running console analysis with default parameters...")
                analyze_configuration_console()
        else:
            # Running as standalone script
            print("Running console analysis with default parameters...")
            analyze_configuration_console()
    except ImportError:
        # Running as standalone script
        print("Running console analysis with default parameters...")
        analyze_configuration_console()
else:
    # Being imported as module - show GUI if available
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            # In Jupyter/Colab environment
            try:
                import ipywidgets as widgets
                from IPython.display import display, clear_output
                create_phm_gui()
                print("✅ Interactive GUI loaded successfully!")
                print("Use the widgets above to analyze different training configurations.")
            except ImportError:
                print("🚀 PHM Training Analyzer imported successfully!")
                print("Use analyze_configuration_console() for analysis")
        else:
            print("🚀 PHM Training Analyzer imported successfully!")
            print("Use analyze_configuration_console() for analysis")
    except ImportError:
        print("🚀 PHM Training Analyzer imported successfully!")
        print("Use analyze_configuration_console() for analysis")
