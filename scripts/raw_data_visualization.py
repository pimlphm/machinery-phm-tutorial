import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt
import seaborn as sns

# === Paths ===
subsets = ['FD001', 'FD002', 'FD003', 'FD004']
columns = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'] + \
          [f'sensor_{i}' for i in range(1, 22)]
sensor_cols = [f'sensor_{i}' for i in range(1, 22)]

# Global variables to store data
global_df = None
current_scaler = None
data_root = None

# === Data root input widget ===
def create_data_root_input():
    style = {'description_width': '150px'}
    layout = widgets.Layout(width='500px')
    
    data_root_input = widgets.Text(
        value="/content/extracted_data/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData",
        placeholder="Enter path to CMAPSS data root directory",
        description='Data Root Path:',
        style=style,
        layout=layout
    )
    
    set_path_btn = widgets.Button(
        description='Set Data Path',
        button_style='primary',
        layout=widgets.Layout(width='150px')
    )
    
    output_area = widgets.Output()
    
    def on_set_path_clicked(b):
        global data_root
        with output_area:
            clear_output(wait=True)
            data_root = data_root_input.value
            print(f"✅ Data root path set to: {data_root}")
            if not os.path.exists(data_root):
                print("⚠️ Warning: Path does not exist!")
    
    set_path_btn.on_click(on_set_path_clicked)
    
    return widgets.VBox([
        widgets.HTML("<h3>📁 Data Path Configuration</h3>"),
        widgets.HBox([data_root_input, set_path_btn]),
        output_area
    ])

# === Load and preprocess function ===
def load_and_preprocess(subset, normalize=True):
    global current_scaler, data_root
    if data_root is None:
        raise ValueError("Data root path not set. Please set the data path first.")
    
    # Load train data
    train_path = os.path.join(data_root, f"train_{subset}.txt")
    rul_path   = os.path.join(data_root, f"RUL_{subset}.txt")
    
    df = pd.read_csv(train_path, sep='\s+', header=None, names=columns)
    rul_df = pd.read_csv(rul_path, sep='\s+', header=None, names=['unit_number', 'RUL_end'])
    
    # Compute RUL for each cycle
    max_cycles = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycle']
    df = df.merge(max_cycles, on='unit_number')
    df['RUL'] = df['max_cycle'] - df['time_in_cycles']
    df.drop('max_cycle', axis=1, inplace=True)
    
    # Normalize sensor data if requested
    if normalize:
        current_scaler = StandardScaler()
        df[sensor_cols] = current_scaler.fit_transform(df[sensor_cols])
    
    return df

# === Sliding window generator ===
def sliding_windows(df, window_size=30, stride=1):
    X, y = [], []
    for unit in df['unit_number'].unique():
        data_unit = df[df['unit_number'] == unit].reset_index(drop=True)
        for start in range(0, len(data_unit) - window_size + 1, stride):
            window = data_unit.iloc[start:start+window_size]
            X.append(window[sensor_cols].values)
            y.append(window['RUL'].values[-1])  # label = RUL at window end
    return np.array(X), np.array(y)

# === Window statistics calculation ===
def calculate_window_statistics(df, window_size=30, stride=5):
    """Calculate statistical features for sliding windows"""
    stats_data = []
    
    for unit in df['unit_number'].unique():
        data_unit = df[df['unit_number'] == unit].reset_index(drop=True)
        for start in range(0, len(data_unit) - window_size + 1, stride):
            window = data_unit.iloc[start:start+window_size]
            window_sensors = window[sensor_cols]
            
            # Calculate statistics for each sensor
            stats_row = {
                'unit_number': unit,
                'window_start': start,
                'window_end': start + window_size - 1,
                'RUL': window['RUL'].iloc[-1],
                'mean_rul': window['RUL'].mean(),
                'time_cycle': window['time_in_cycles'].iloc[-1]
            }
            
            # Add sensor statistics
            for sensor in sensor_cols:
                sensor_data = window_sensors[sensor]
                stats_row.update({
                    f'{sensor}_mean': sensor_data.mean(),
                    f'{sensor}_std': sensor_data.std(),
                    f'{sensor}_min': sensor_data.min(),
                    f'{sensor}_max': sensor_data.max(),
                    f'{sensor}_range': sensor_data.max() - sensor_data.min(),
                    f'{sensor}_trend': np.polyfit(range(len(sensor_data)), sensor_data, 1)[0] if len(sensor_data) > 1 else 0
                })
            
            stats_data.append(stats_row)
    
    return pd.DataFrame(stats_data)

# === Dataset class ===
class CMapssDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === Dashboard visualization ===
def create_dashboard_plot(df, window_stats, sensor_name, unit_numbers):
    """Create comprehensive dashboard with original data and window statistics"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Dashboard: {sensor_name} Analysis with Window Statistics', fontsize=16)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(unit_numbers)))
    
    for i, unit in enumerate(unit_numbers):
        unit_data = df[df['unit_number'] == unit]
        unit_stats = window_stats[window_stats['unit_number'] == unit]
        color = colors[i]
        
        # 1. Original sensor data
        axes[0, 0].plot(unit_data['time_in_cycles'], unit_data[sensor_name], 
                       color=color, label=f'Unit {unit}', linewidth=1.5)
        
        # 2. RUL progression
        axes[0, 1].plot(unit_data['time_in_cycles'], unit_data['RUL'], 
                       color=color, label=f'Unit {unit}', linewidth=1.5)
        
        # 3. Window mean vs RUL
        axes[0, 2].scatter(unit_stats['RUL'], unit_stats[f'{sensor_name}_mean'], 
                          color=color, alpha=0.6, label=f'Unit {unit}', s=20)
        
        # 4. Window std vs RUL
        axes[1, 0].scatter(unit_stats['RUL'], unit_stats[f'{sensor_name}_std'], 
                          color=color, alpha=0.6, label=f'Unit {unit}', s=20)
        
        # 5. Window trend vs RUL
        axes[1, 1].scatter(unit_stats['RUL'], unit_stats[f'{sensor_name}_trend'], 
                          color=color, alpha=0.6, label=f'Unit {unit}', s=20)
        
        # 6. Window range vs RUL
        axes[1, 2].scatter(unit_stats['RUL'], unit_stats[f'{sensor_name}_range'], 
                          color=color, alpha=0.6, label=f'Unit {unit}', s=20)
    
    # Set titles and labels
    axes[0, 0].set_title(f'{sensor_name} Time Series')
    axes[0, 0].set_xlabel('Time in Cycles')
    axes[0, 0].set_ylabel(sensor_name)
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('RUL Progression')
    axes[0, 1].set_xlabel('Time in Cycles')
    axes[0, 1].set_ylabel('RUL')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_title(f'Window Mean vs RUL')
    axes[0, 2].set_xlabel('RUL')
    axes[0, 2].set_ylabel(f'{sensor_name} Mean')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].set_title(f'Window Std vs RUL')
    axes[1, 0].set_xlabel('RUL')
    axes[1, 0].set_ylabel(f'{sensor_name} Std')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title(f'Window Trend vs RUL')
    axes[1, 1].set_xlabel('RUL')
    axes[1, 1].set_ylabel(f'{sensor_name} Trend')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].set_title(f'Window Range vs RUL')
    axes[1, 2].set_xlabel('RUL')
    axes[1, 2].set_ylabel(f'{sensor_name} Range')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# === Raw data visualization ===
def create_raw_data_plot(df, sensor_names, unit_numbers, max_units_per_plot=10):
    """Create comprehensive visualization of raw sensor data"""
    
    if len(sensor_names) == 1:
        # Single sensor analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Raw Data Analysis: {sensor_names[0]}', fontsize=16)
        
        sensor = sensor_names[0]
        colors = plt.cm.Set1(np.linspace(0, 1, min(len(unit_numbers), max_units_per_plot)))
        
        for i, unit in enumerate(unit_numbers[:max_units_per_plot]):
            unit_data = df[df['unit_number'] == unit]
            color = colors[i]
            
            # Time series plot
            axes[0, 0].plot(unit_data['time_in_cycles'], unit_data[sensor], 
                           color=color, label=f'Unit {unit}', linewidth=1.2, alpha=0.8)
            
            # RUL vs sensor value
            axes[0, 1].scatter(unit_data['RUL'], unit_data[sensor], 
                              color=color, alpha=0.6, label=f'Unit {unit}', s=15)
            
            # Cycle vs sensor (normalized by max cycle)
            normalized_cycle = unit_data['time_in_cycles'] / unit_data['time_in_cycles'].max()
            axes[1, 0].scatter(normalized_cycle, unit_data[sensor], 
                              color=color, alpha=0.6, label=f'Unit {unit}', s=15)
            
            # Distribution of sensor values
            axes[1, 1].hist(unit_data[sensor], bins=20, alpha=0.5, 
                           color=color, label=f'Unit {unit}', density=True)
        
        axes[0, 0].set_title(f'{sensor} vs Time')
        axes[0, 0].set_xlabel('Time in Cycles')
        axes[0, 0].set_ylabel(sensor)
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title(f'{sensor} vs RUL')
        axes[0, 1].set_xlabel('RUL')
        axes[0, 1].set_ylabel(sensor)
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title(f'{sensor} vs Normalized Cycle')
        axes[1, 0].set_xlabel('Normalized Time (0-1)')
        axes[1, 0].set_ylabel(sensor)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title(f'{sensor} Distribution')
        axes[1, 1].set_xlabel(sensor)
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
    else:
        # Multiple sensor comparison
        n_sensors = len(sensor_names)
        n_cols = min(4, n_sensors)
        n_rows = (n_sensors + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'Raw Data Comparison: Multiple Sensors', fontsize=16)
        
        colors = plt.cm.Set1(np.linspace(0, 1, min(len(unit_numbers), max_units_per_plot)))
        
        for i, sensor in enumerate(sensor_names):
            ax = axes[i] if n_sensors > 1 else axes[0]
            
            for j, unit in enumerate(unit_numbers[:max_units_per_plot]):
                unit_data = df[df['unit_number'] == unit]
                color = colors[j]
                
                ax.plot(unit_data['time_in_cycles'], unit_data[sensor], 
                       color=color, label=f'Unit {unit}', linewidth=1.0, alpha=0.7)
            
            ax.set_title(f'{sensor} Time Series')
            ax.set_xlabel('Time in Cycles')
            ax.set_ylabel(sensor)
            if i == 0:  # Only show legend for first subplot
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_sensors, len(axes)):
            axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# === Interactive Parameter Configuration Widget ===
def create_data_config_widget():
    global global_df
    
    # Style definition
    style = {'description_width': '150px'}
    layout = widgets.Layout(width='300px')
    
    # Widget components
    subset_selector = widgets.SelectMultiple(
        options=subsets,
        value=['FD001'],
        description='Select Subsets:',
        style=style,
        layout=layout
    )
    
    window_size_slider = widgets.IntSlider(
        value=30,
        min=10,
        max=100,
        step=5,
        description='Window Size:',
        style=style,
        layout=layout
    )
    
    stride_slider = widgets.IntSlider(
        value=5,
        min=1,
        max=20,
        step=1,
        description='Stride:',
        style=style,
        layout=layout
    )
    
    normalize_checkbox = widgets.Checkbox(
        value=True,
        description='Normalize Data',
        style=style
    )
    
    train_split_slider = widgets.FloatSlider(
        value=0.7,
        min=0.5,
        max=0.8,
        step=0.05,
        description='Train Split:',
        style=style,
        layout=layout
    )
    
    val_split_slider = widgets.FloatSlider(
        value=0.15,
        min=0.1,
        max=0.3,
        step=0.05,
        description='Val Split:',
        style=style,
        layout=layout
    )
    
    batch_size_dropdown = widgets.Dropdown(
        options=[16, 32, 64, 128, 256],
        value=64,
        description='Batch Size:',
        style=style,
        layout=layout
    )
    
    # Visualization components
    sensor_dropdown = widgets.Dropdown(
        options=sensor_cols,
        value='sensor_1',
        description='Select Sensor:',
        style=style,
        layout=layout
    )
    
    multi_sensor_selector = widgets.SelectMultiple(
        options=sensor_cols,
        value=['sensor_1'],
        description='Multiple Sensors:',
        style=style,
        layout=widgets.Layout(width='300px', height='100px')
    )
    
    unit_selector = widgets.SelectMultiple(
        options=[],
        value=[],
        description='Select Units:',
        style=style,
        layout=widgets.Layout(width='300px', height='100px')
    )
    
    plot_type_toggle = widgets.ToggleButtons(
        options=[('Dashboard', 'dashboard'), ('Raw Data', 'raw')],
        value='dashboard',
        description='Plot Type:',
        style=style
    )
    
    # Buttons
    load_data_btn = widgets.Button(
        description='Load Data',
        button_style='primary',
        layout=widgets.Layout(width='200px')
    )
    
    create_dataset_btn = widgets.Button(
        description='Create DataLoaders',
        button_style='success',
        layout=widgets.Layout(width='200px')
    )
    
    plot_data_btn = widgets.Button(
        description='Generate Plot',
        button_style='info',
        layout=widgets.Layout(width='200px')
    )
    
    # Output areas
    output_area = widgets.Output()
    plot_output = widgets.Output()
    
    # Event handlers
    def on_load_data_clicked(b):
        with output_area:
            clear_output(wait=True)
            if data_root is None:
                print("❌ Please set data root path first!")
                return
                
            print("Loading data...")
            
            try:
                global global_df
                all_dfs = []
                for subset in subset_selector.value:
                    df = load_and_preprocess(subset, normalize_checkbox.value)
                    all_dfs.append(df)
                
                global_df = pd.concat(all_dfs, ignore_index=True)
                
                # Update unit selector options
                available_units = sorted(global_df['unit_number'].unique())
                unit_selector.options = available_units
                unit_selector.value = available_units[:min(5, len(available_units))]
                
                print(f"✅ Data loaded successfully!")
                print(f"   Total samples: {len(global_df)}")
                print(f"   Number of units: {len(available_units)}")
                print(f"   Selected subsets: {list(subset_selector.value)}")
                print(f"   Data normalization: {'Enabled' if normalize_checkbox.value else 'Disabled'}")
                
            except Exception as e:
                print(f"❌ Error loading data: {str(e)}")
    
    def on_create_dataset_clicked(b):
        with output_area:
            if global_df is None:
                print("❌ Please load data first!")
                return
            
            print("Creating datasets and dataloaders...")
            
            try:
                # Create sliding windows
                X, y = sliding_windows(global_df, window_size_slider.value, stride_slider.value)
                
                # Split data
                n = len(X)
                idx = np.random.permutation(n)
                train_end = int(train_split_slider.value * n)
                val_end = int((train_split_slider.value + val_split_slider.value) * n)
                
                train_idx = idx[:train_end]
                val_idx = idx[train_end:val_end]
                test_idx = idx[val_end:]
                
                # Create datasets
                train_ds = CMapssDataset(X[train_idx], y[train_idx])
                val_ds = CMapssDataset(X[val_idx], y[val_idx])
                test_ds = CMapssDataset(X[test_idx], y[test_idx])
                
                # Create dataloaders
                global train_loader, val_loader, test_loader
                train_loader = DataLoader(train_ds, batch_size=batch_size_dropdown.value, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=batch_size_dropdown.value, shuffle=False)
                test_loader = DataLoader(test_ds, batch_size=batch_size_dropdown.value, shuffle=False)
                
                print("✅ DataLoaders created successfully!")
                print(f"   Window size: {window_size_slider.value}")
                print(f"   Stride: {stride_slider.value}")
                print(f"   Train samples: {len(train_ds)}")
                print(f"   Validation samples: {len(val_ds)}")
                print(f"   Test samples: {len(test_ds)}")
                print(f"   Batch size: {batch_size_dropdown.value}")
                print(f"   Train split: {train_split_slider.value:.1%}")
                print(f"   Val split: {val_split_slider.value:.1%}")
                print(f"   Test split: {1 - train_split_slider.value - val_split_slider.value:.1%}")
                
            except Exception as e:
                print(f"❌ Error creating datasets: {str(e)}")
    
    def on_plot_data_clicked(b):
        with plot_output:
            clear_output(wait=True)
            if global_df is None:
                print("❌ Please load data first!")
                return
            
            if len(unit_selector.value) == 0:
                print("❌ Please select at least one unit!")
                return
            
            try:
                selected_units = list(unit_selector.value)
                
                if plot_type_toggle.value == 'dashboard':
                    print("🚀 Calculating window statistics...")
                    # Calculate window statistics
                    window_stats = calculate_window_statistics(
                        global_df, 
                        window_size=window_size_slider.value, 
                        stride=stride_slider.value
                    )
                    
                    print("🚀 Generating dashboard...")
                    create_dashboard_plot(
                        global_df, 
                        window_stats,
                        sensor_dropdown.value, 
                        selected_units
                    )
                else:  # Raw data plot
                    print("🚀 Generating raw data visualization...")
                    selected_sensors = list(multi_sensor_selector.value)
                    create_raw_data_plot(
                        global_df,
                        selected_sensors,
                        selected_units
                    )
                    
            except Exception as e:
                print(f"❌ Error plotting data: {str(e)}")
    
    # Connect event handlers
    load_data_btn.on_click(on_load_data_clicked)
    create_dataset_btn.on_click(on_create_dataset_clicked)
    plot_data_btn.on_click(on_plot_data_clicked)
    
    # Layout
    config_box = widgets.VBox([
        widgets.HTML("<h3>📊 CMAPSS Data Configuration</h3>"),
        widgets.HBox([
            widgets.VBox([
                widgets.HTML("<h4>Data Loading Parameters</h4>"),
                subset_selector,
                normalize_checkbox,
            ]),
            widgets.VBox([
                widgets.HTML("<h4>Window Parameters</h4>"),
                window_size_slider,
                stride_slider,
            ]),
            widgets.VBox([
                widgets.HTML("<h4>Split Parameters</h4>"),
                train_split_slider,
                val_split_slider,
                batch_size_dropdown,
            ])
        ]),
        widgets.HBox([load_data_btn, create_dataset_btn]),
        output_area
    ])
    
    viz_box = widgets.VBox([
        widgets.HTML("<h3>📈 Data Visualization</h3>"),
        plot_type_toggle,
        widgets.HBox([
            widgets.VBox([
                sensor_dropdown,
                widgets.HTML("<small>For Dashboard plot</small>")
            ]),
            widgets.VBox([
                multi_sensor_selector,
                widgets.HTML("<small>For Raw Data plot</small>")
            ])
        ]),
        unit_selector,
        plot_data_btn,
        widgets.HTML("<small>💡 Dashboard: Time series, RUL progression, and window statistics | Raw Data: Original sensor data analysis</small>"),
        plot_output
    ])
    
    main_widget = widgets.Tab([config_box, viz_box])
    main_widget.set_title(0, 'Configuration')
    main_widget.set_title(1, 'Visualization')
    
    return main_widget

# === Main function ===
def raw_data_visualization():
    """Main function to initialize and display the CMAPSS raw data visualization interface"""
    print("🚀 Initializing CMAPSS Raw Data Visualization Tool")
    print("=" * 60)
    
    # Create and display data root input
    path_widget = create_data_root_input()
    display(path_widget)
    
    # Create and display main configuration widget
    config_widget = create_data_config_widget()
    display(config_widget)
    
    print("\n📋 Instructions:")
    print("1. First, set the data root path pointing to your CMAPSS data directory")
    print("2. Configure data loading parameters in the Configuration tab")
    print("3. Load data and create datasets as needed")
    print("4. Use the Visualization tab to plot data:")
    print("   - Dashboard: Shows window statistics analysis")
    print("   - Raw Data: Shows original sensor data visualization")

if __name__ == "__main__":
    raw_data_visualization()
