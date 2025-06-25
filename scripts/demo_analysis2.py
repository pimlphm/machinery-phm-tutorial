import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class TurbofanAnalysis:
    def __init__(self):
        # Data storage
        self.datasets = None
        self.combined_train = None
        self.combined_test = None
        self.selected_sensors = []
        self.model_without_preprocessing = None
        self.model_with_preprocessing = None
        self.results_without_preprocess = None
        self.results_with_preprocess = None
        
        # Store fitted preprocessors
        self.scaler = None
        
        # Sensor descriptions for user selection
        self.sensor_descriptions = {
            'sensor_1': 'Total temperature at fan inlet (°R)',
            'sensor_2': 'Total temperature at LPC outlet (°R)',
            'sensor_3': 'Total temperature at HPC outlet (°R)',
            'sensor_4': 'Total temperature at LPT outlet (°R)',
            'sensor_5': 'Pressure at fan inlet (psia)',
            'sensor_6': 'Total pressure in bypass-duct (psia)',
            'sensor_7': 'Total pressure at HPC outlet (psia)',
            'sensor_8': 'Physical fan speed (rpm)',
            'sensor_9': 'Physical core speed (rpm)',
            'sensor_10': 'Engine pressure ratio (P50/P2)',
            'sensor_11': 'Static pressure at HPC outlet (psia)',
            'sensor_12': 'Ratio of fuel flow to Ps30 (pps/psi)',
            'sensor_13': 'Corrected fan speed (rpm)',
            'sensor_14': 'Corrected core speed (rpm)',
            'sensor_15': 'Bypass Ratio',
            'sensor_16': 'Burner fuel-air ratio',
            'sensor_17': 'Bleed Enthalpy',
            'sensor_18': 'Required fan speed (rpm)',
            'sensor_19': 'Required core speed (rpm)',
            'sensor_20': 'High-pressure turbine coolant bleed (lbm/s)',
            'sensor_21': 'Low-pressure turbine coolant bleed (lbm/s)'
        }
        
        # Recommended sensor combinations from literature
        self.recommended_combinations = {
            'Literature_Set_1': ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'],
            'Literature_Set_2': ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'],
            'High_Variance_Set': ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'],
            'Correlation_Based': ['sensor_3', 'sensor_4', 'sensor_7', 'sensor_11', 'sensor_12', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'],
            'Physical_Meaningful': ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_8', 'sensor_9', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17']
        }
        
    def load_cmapss_data(self, base_path):
        """Load all CMAPSS training and test data"""
        columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor_{i}' for i in range(1, 22)]
        datasets = {}
        
        for i in range(1, 5):
            train_file = f"{base_path}/train_FD00{i}.txt"
            test_file = f"{base_path}/test_FD00{i}.txt"
            rul_file = f"{base_path}/RUL_FD00{i}.txt"
            
            try:
                # Load training data
                train_data = pd.read_csv(train_file, sep='\s+', header=None, names=columns)
                
                # Calculate RUL for training data
                max_cycles = train_data.groupby('engine_id')['cycle'].max().reset_index()
                max_cycles.columns = ['engine_id', 'max_cycle']
                train_data = train_data.merge(max_cycles, on='engine_id')
                train_data['RUL'] = train_data['max_cycle'] - train_data['cycle']
                train_data = train_data.drop('max_cycle', axis=1)
                
                # Load test data
                test_data = pd.read_csv(test_file, sep='\s+', header=None, names=columns)
                
                # Load true RUL values for test data
                true_rul = pd.read_csv(rul_file, sep='\s+', header=None, names=['true_RUL'])
                
                # Calculate RUL for test data
                test_max_cycles = test_data.groupby('engine_id')['cycle'].max().reset_index()
                test_max_cycles.columns = ['engine_id', 'max_cycle']
                test_max_cycles['true_RUL'] = true_rul['true_RUL'].values
                test_max_cycles['total_cycles'] = test_max_cycles['max_cycle'] + test_max_cycles['true_RUL']
                
                test_data = test_data.merge(test_max_cycles[['engine_id', 'total_cycles']], on='engine_id')
                test_data['RUL'] = test_data['total_cycles'] - test_data['cycle']
                test_data = test_data.drop('total_cycles', axis=1)
                
                datasets[f'FD00{i}'] = {
                    'train': train_data,
                    'test': test_data
                }
                
                print(f"✅ Loaded FD00{i}: Train={len(train_data)} samples, Test={len(test_data)} samples")
                
            except Exception as e:
                print(f"❌ Error loading FD00{i}: {str(e)}")
        
        return datasets
    
    def combine_datasets(self):
        """Combine all datasets"""
        all_train_data = []
        all_test_data = []
        
        for dataset_name, data in self.datasets.items():
            train_data = data['train'].copy()
            test_data = data['test'].copy()
            
            # Add dataset identifier
            train_data['dataset'] = dataset_name
            test_data['dataset'] = dataset_name
            
            all_train_data.append(train_data)
            all_test_data.append(test_data)
        
        # Combine all data
        self.combined_train = pd.concat(all_train_data, ignore_index=True)
        self.combined_test = pd.concat(all_test_data, ignore_index=True)
        
        print(f"📈 Combined Training Data: {len(self.combined_train)} samples")
        print(f"📉 Combined Test Data: {len(self.combined_test)} samples")
    
    def display_sensor_options(self):
        """Display all available sensors with descriptions for user selection"""
        print("\n🔧 Available Sensors for Selection:")
        print("="*80)
        for sensor_id, description in self.sensor_descriptions.items():
            print(f"  {sensor_id}: {description}")
        print("="*80)
    
    def display_recommended_combinations(self):
        """Display recommended sensor combinations from literature"""
        print("\n📚 Recommended Sensor Combinations from Literature:")
        print("="*80)
        for combo_name, sensors in self.recommended_combinations.items():
            print(f"\n🎯 {combo_name}:")
            print(f"   Sensors: {', '.join(sensors)}")
            print(f"   Count: {len(sensors)} sensors")
            print("   Descriptions:")
            for sensor in sensors:
                print(f"     - {sensor}: {self.sensor_descriptions[sensor]}")
        print("="*80)
    
    def get_user_selected_sensors(self):
        """Interactive sensor selection with checkbox-like interface"""
        self.display_sensor_options()
        self.display_recommended_combinations()
        
        print("\n📝 Sensor Selection Options:")
        print("1. Select individual sensors (enter numbers)")
        print("2. Use recommended combination")
        print("3. Auto-select based on variance")
        
        choice = input("Choose option (1/2/3): ").strip()
        
        if choice == "1":
            return self._manual_sensor_selection()
        elif choice == "2":
            return self._recommended_sensor_selection()
        elif choice == "3":
            return self.auto_select_sensors()
        else:
            print("❌ Invalid choice. Using auto-selection...")
            return self.auto_select_sensors()
    
    def _manual_sensor_selection(self):
        """Manual sensor selection with checkbox-like interface"""
        print("\n🔧 Manual Sensor Selection:")
        print("Enter sensor numbers (1-21) you want to SELECT, separated by commas")
        print("Example: 1,2,3,8,9")
        
        selected_sensors = []
        
        # Show all sensors for selection
        print("\nAvailable sensors:")
        for i in range(1, 22):
            sensor_name = f'sensor_{i}'
            print(f"  {i}: {self.sensor_descriptions[sensor_name]}")
        
        user_input = input("\nEnter sensor numbers to select: ").strip()
        
        try:
            # Parse user input
            sensor_numbers = [int(x.strip()) for x in user_input.split(',')]
            
            for num in sensor_numbers:
                if 1 <= num <= 21:
                    sensor_name = f'sensor_{num}'
                    selected_sensors.append(sensor_name)
                    print(f"✅ Selected: {sensor_name} - {self.sensor_descriptions[sensor_name]}")
                else:
                    print(f"⚠️ Invalid sensor number: {num} (must be 1-21)")
            
            if selected_sensors:
                print(f"\n🎯 Total selected: {len(selected_sensors)} sensors")
                return selected_sensors
            else:
                print("❌ No valid sensors selected. Using auto-selection...")
                return self.auto_select_sensors()
                
        except ValueError:
            print("❌ Invalid input format. Using auto-selection...")
            return self.auto_select_sensors()
    
    def _recommended_sensor_selection(self):
        """Select from recommended combinations"""
        print("\n📚 Choose a recommended sensor combination:")
        
        combinations = list(self.recommended_combinations.keys())
        for i, combo_name in enumerate(combinations, 1):
            sensors = self.recommended_combinations[combo_name]
            print(f"{i}. {combo_name} ({len(sensors)} sensors)")
        
        try:
            choice = int(input(f"\nEnter choice (1-{len(combinations)}): ").strip())
            if 1 <= choice <= len(combinations):
                selected_combo = combinations[choice - 1]
                selected_sensors = self.recommended_combinations[selected_combo]
                
                print(f"\n✅ Selected: {selected_combo}")
                print("Selected sensors:")
                for sensor in selected_sensors:
                    print(f"  {sensor}: {self.sensor_descriptions[sensor]}")
                
                return selected_sensors
            else:
                print("❌ Invalid choice. Using auto-selection...")
                return self.auto_select_sensors()
        except ValueError:
            print("❌ Invalid input. Using auto-selection...")
            return self.auto_select_sensors()
    
    def auto_select_sensors(self, variance_threshold=0.1):
        """Auto select sensors with high variance"""
        if self.combined_train is None:
            print("⚠️ Please load data first!")
            return []
        
        # Calculate variance for each sensor
        sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
        variances = self.combined_train[sensor_cols].var()
        
        # Select sensors with variance > threshold
        selected_sensors = variances[variances > variance_threshold].index.tolist()
        
        print(f"\n🔍 Auto selected {len(selected_sensors)} sensors with variance > {variance_threshold}")
        print("Selected sensors:")
        for sensor in selected_sensors:
            print(f"  {sensor}: {self.sensor_descriptions[sensor]}")
        
        return selected_sensors
    
    def create_simple_features(self, data, is_training=True):
        """
        简单高效的特征工程，确保大幅提升性能
        """
        processed_data = data.copy()
        sensor_cols = [col for col in self.selected_sensors if col in data.columns]
        
        print(f"🔧 Creating simple effective features...")
        
        # 1. 滚动统计特征 (非常有效的时间序列特征)
        window_sizes = [3, 5]
        
        for engine_id in processed_data['engine_id'].unique():
            engine_mask = processed_data['engine_id'] == engine_id
            engine_data = processed_data[engine_mask].sort_values('cycle').copy()
            
            for sensor in sensor_cols:
                for window in window_sizes:
                    # 滚动均值
                    rolling_mean = engine_data[sensor].rolling(window=window, min_periods=1).mean()
                    processed_data.loc[engine_mask, f'{sensor}_rolling_mean_{window}'] = rolling_mean.values
                    
                    # 滚动标准差
                    rolling_std = engine_data[sensor].rolling(window=window, min_periods=1).std().fillna(0)
                    processed_data.loc[engine_mask, f'{sensor}_rolling_std_{window}'] = rolling_std.values
            
            # 2. 累积退化特征 (对RUL预测很重要)
            for sensor in sensor_cols:
                # 累积均值偏差 (相对于初始值的偏差)
                initial_value = engine_data[sensor].iloc[0]
                cumulative_deviation = (engine_data[sensor] - initial_value) / (abs(initial_value) + 1e-6)
                processed_data.loc[engine_mask, f'{sensor}_cumulative_deviation'] = cumulative_deviation.values
                
                # 趋势特征 (线性回归斜率)
                cycles = np.arange(len(engine_data))
                if len(cycles) > 1:
                    slope = np.polyfit(cycles, engine_data[sensor].values, 1)[0]
                    processed_data.loc[engine_mask, f'{sensor}_trend'] = slope
                else:
                    processed_data.loc[engine_mask, f'{sensor}_trend'] = 0
        
        # 3. 传感器比值特征 (物理相关性)
        important_ratios = [
            ('sensor_3', 'sensor_2'),  # 温度比
            ('sensor_9', 'sensor_8'),  # 转速比
            ('sensor_7', 'sensor_5'),  # 压力比
        ]
        
        for sensor1, sensor2 in important_ratios:
            if sensor1 in sensor_cols and sensor2 in sensor_cols:
                ratio_col = f'{sensor1}_{sensor2}_ratio'
                processed_data[ratio_col] = processed_data[sensor1] / (processed_data[sensor2] + 1e-6)
        
        # 4. 简单周期特征
        processed_data['cycle_squared'] = processed_data['cycle'] ** 2
        processed_data['cycle_log'] = np.log(processed_data['cycle'] + 1)
        
        print(f"✅ Created simple effective features")
        return processed_data
    
    def train_models(self, interactive_selection=True):
        """
        训练基线模型和预处理模型进行对比
        """
        if self.combined_train is None or self.combined_test is None:
            print("⚠️ Please load and combine datasets first!")
            return None, None
        
        # Sensor selection
        if interactive_selection:
            self.selected_sensors = self.get_user_selected_sensors()
        else:
            self.selected_sensors = self.auto_select_sensors()
        
        if not self.selected_sensors:
            print("❌ No sensors selected!")
            return None, None
        
        print(f"\n🎯 Using {len(self.selected_sensors)} sensors: {self.selected_sensors}")
        
        # Prepare baseline data (without feature engineering)
        print("\n" + "="*50)
        print("🔧 PREPARING BASELINE DATA")
        print("="*50)
        
        baseline_train = self.combined_train.copy()
        baseline_test = self.combined_test.copy()
        
        # Only keep selected features for baseline
        baseline_X_train = baseline_train[self.selected_sensors + ['cycle']]
        baseline_y_train = baseline_train['RUL']
        baseline_X_test = baseline_test[self.selected_sensors + ['cycle']]
        baseline_y_test = baseline_test['RUL']
        
        # Prepare enhanced data (with feature engineering)
        print("\n" + "="*50)
        print("🔧 PREPARING ENHANCED DATA WITH SIMPLE FEATURE ENGINEERING")
        print("="*50)
        
        enhanced_train = self.create_simple_features(self.combined_train.copy(), is_training=True)
        enhanced_test = self.create_simple_features(self.combined_test.copy(), is_training=False)
        
        # Get feature columns (exclude non-feature columns)
        exclude_cols = ['engine_id', 'setting1', 'setting2', 'setting3', 'dataset', 'RUL']
        enhanced_feature_cols = [col for col in enhanced_train.columns if col not in exclude_cols]
        
        enhanced_X_train = enhanced_train[enhanced_feature_cols]
        enhanced_y_train = enhanced_train['RUL']
        enhanced_X_test = enhanced_test[enhanced_feature_cols]
        enhanced_y_test = enhanced_test['RUL']
        
        # 标准化特征
        self.scaler = StandardScaler()
        enhanced_X_train_scaled = self.scaler.fit_transform(enhanced_X_train)
        enhanced_X_test_scaled = self.scaler.transform(enhanced_X_test)
        
        # Convert back to DataFrame
        enhanced_X_train_scaled = pd.DataFrame(enhanced_X_train_scaled, columns=enhanced_feature_cols, index=enhanced_X_train.index)
        enhanced_X_test_scaled = pd.DataFrame(enhanced_X_test_scaled, columns=enhanced_feature_cols, index=enhanced_X_test.index)
        
        print(f"📊 Baseline features: {len(baseline_X_train.columns)}")
        print(f"📊 Enhanced features: {len(enhanced_X_train.columns)}")
        
        # Train baseline model
        print("\n" + "="*50)
        print("🤖 TRAINING BASELINE MODEL")
        print("="*50)
        
        self.model_without_preprocessing = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model_without_preprocessing.fit(baseline_X_train, baseline_y_train)
        
        # Predictions and metrics for baseline model
        baseline_y_pred_train = self.model_without_preprocessing.predict(baseline_X_train)
        baseline_y_pred_test = self.model_without_preprocessing.predict(baseline_X_test)
        
        self.results_without_preprocess = {
            'train_rmse': np.sqrt(mean_squared_error(baseline_y_train, baseline_y_pred_train)),
            'train_mae': mean_absolute_error(baseline_y_train, baseline_y_pred_train),
            'train_r2': r2_score(baseline_y_train, baseline_y_pred_train),
            'test_rmse': np.sqrt(mean_squared_error(baseline_y_test, baseline_y_pred_test)),
            'test_mae': mean_absolute_error(baseline_y_test, baseline_y_pred_test),
            'test_r2': r2_score(baseline_y_test, baseline_y_pred_test),
            'y_test': baseline_y_test,
            'y_pred_test': baseline_y_pred_test,
            'feature_importance': dict(zip(baseline_X_train.columns, self.model_without_preprocessing.feature_importances_))
        }
        
        # Train enhanced model
        print("\n" + "="*50)
        print("🤖 TRAINING ENHANCED MODEL WITH FEATURE ENGINEERING")
        print("="*50)
        
        self.model_with_preprocessing = RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        
        self.model_with_preprocessing.fit(enhanced_X_train_scaled, enhanced_y_train)
        
        # Predictions and metrics for enhanced model
        enhanced_y_pred_train = self.model_with_preprocessing.predict(enhanced_X_train_scaled)
        enhanced_y_pred_test = self.model_with_preprocessing.predict(enhanced_X_test_scaled)
        
        self.results_with_preprocess = {
            'train_rmse': np.sqrt(mean_squared_error(enhanced_y_train, enhanced_y_pred_train)),
            'train_mae': mean_absolute_error(enhanced_y_train, enhanced_y_pred_train),
            'train_r2': r2_score(enhanced_y_train, enhanced_y_pred_train),
            'test_rmse': np.sqrt(mean_squared_error(enhanced_y_test, enhanced_y_pred_test)),
            'test_mae': mean_absolute_error(enhanced_y_test, enhanced_y_pred_test),
            'test_r2': r2_score(enhanced_y_test, enhanced_y_pred_test),
            'y_test': enhanced_y_test,
            'y_pred_test': enhanced_y_pred_test,
            'feature_importance': dict(zip(enhanced_feature_cols, self.model_with_preprocessing.feature_importances_))
        }
        
        # Print results comparison
        self.print_results_comparison()
        
        return self.results_without_preprocess, self.results_with_preprocess
    
    def print_results_comparison(self):
        """Print comparison of results"""
        print("\n" + "="*80)
        print("📊 MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        print("\n🔍 BASELINE MODEL (No Preprocessing):")
        print(f"  Train RMSE: {self.results_without_preprocess['train_rmse']:.2f}")
        print(f"  Train MAE:  {self.results_without_preprocess['train_mae']:.2f}")
        print(f"  Train R²:   {self.results_without_preprocess['train_r2']:.4f}")
        print(f"  Test RMSE:  {self.results_without_preprocess['test_rmse']:.2f}")
        print(f"  Test MAE:   {self.results_without_preprocess['test_mae']:.2f}")
        print(f"  Test R²:    {self.results_without_preprocess['test_r2']:.4f}")
        
        print("\n🚀 ENHANCED MODEL (With Feature Engineering):")
        print(f"  Train RMSE: {self.results_with_preprocess['train_rmse']:.2f}")
        print(f"  Train MAE:  {self.results_with_preprocess['train_mae']:.2f}")
        print(f"  Train R²:   {self.results_with_preprocess['train_r2']:.4f}")
        print(f"  Test RMSE:  {self.results_with_preprocess['test_rmse']:.2f}")
        print(f"  Test MAE:   {self.results_with_preprocess['test_mae']:.2f}")
        print(f"  Test R²:    {self.results_with_preprocess['test_r2']:.4f}")
        
        # Calculate improvements
        rmse_improvement = ((self.results_without_preprocess['test_rmse'] - self.results_with_preprocess['test_rmse']) / self.results_without_preprocess['test_rmse']) * 100
        mae_improvement = ((self.results_without_preprocess['test_mae'] - self.results_with_preprocess['test_mae']) / self.results_without_preprocess['test_mae']) * 100
        r2_improvement = ((self.results_with_preprocess['test_r2'] - self.results_without_preprocess['test_r2']) / abs(self.results_without_preprocess['test_r2'])) * 100
        
        print("\n📈 IMPROVEMENTS:")
        print(f"  RMSE Reduction: {rmse_improvement:.1f}%")
        print(f"  MAE Reduction:  {mae_improvement:.1f}%")
        print(f"  R² Improvement: {r2_improvement:.1f}%")
        
        if rmse_improvement > 0 and mae_improvement > 0:
            print("\n✅ Feature engineering successfully improved model performance!")
        else:
            print("\n⚠️ Feature engineering did not improve performance as expected.")
    
    def create_comparison_visualization(self):
        """Create comprehensive comparison visualization"""
        if self.results_without_preprocess is None or self.results_with_preprocess is None:
            print("⚠️ No results to visualize!")
            return
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Model Comparison: Baseline vs Feature Engineering\nUsing {len(self.selected_sensors)} Sensors', 
                     fontsize=16, fontweight='bold')
        
        # Sample data for visualization
        sample_size = min(2000, len(self.results_without_preprocess['y_test']))
        sample_indices = np.random.choice(len(self.results_without_preprocess['y_test']), sample_size, replace=False)
        
        # Plot 1: Prediction vs True (Baseline)
        ax1 = axes[0, 0]
        y_test = self.results_without_preprocess['y_test']
        y_pred_test = self.results_without_preprocess['y_pred_test']
        ax1.scatter(y_test.iloc[sample_indices], y_pred_test[sample_indices], alpha=0.6, s=15, color='blue')
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', alpha=0.8, linewidth=2)
        ax1.set_xlabel('True RUL Values')
        ax1.set_ylabel('Predicted RUL Values')
        ax1.set_title('Baseline Model Results')
        ax1.grid(True, alpha=0.3)
        ax1.text(0.05, 0.95, f'R² = {self.results_without_preprocess["test_r2"]:.3f}\nRMSE = {self.results_without_preprocess["test_rmse"]:.1f}', 
                transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))
        
        # Plot 2: Prediction vs True (Enhanced)
        ax2 = axes[0, 1]
        y_test_enh = self.results_with_preprocess['y_test']
        y_pred_test_enh = self.results_with_preprocess['y_pred_test']
        ax2.scatter(y_test_enh.iloc[sample_indices], y_pred_test_enh[sample_indices], alpha=0.6, s=15, color='green')
        ax2.plot([y_test_enh.min(), y_test_enh.max()], [y_test_enh.min(), y_test_enh.max()], 'r--', alpha=0.8, linewidth=2)
        ax2.set_xlabel('True RUL Values')
        ax2.set_ylabel('Predicted RUL Values')
        ax2.set_title('Enhanced Model Results')
        ax2.grid(True, alpha=0.3)
        ax2.text(0.05, 0.95, f'R² = {self.results_with_preprocess["test_r2"]:.3f}\nRMSE = {self.results_with_preprocess["test_rmse"]:.1f}', 
                transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))
        
        # Plot 3: Metrics Comparison
        ax3 = axes[0, 2]
        metrics = ['RMSE', 'MAE', 'R²']
        baseline_values = [self.results_without_preprocess['test_rmse'], 
                          self.results_without_preprocess['test_mae'], 
                          self.results_without_preprocess['test_r2']]
        enhanced_values = [self.results_with_preprocess['test_rmse'], 
                          self.results_with_preprocess['test_mae'], 
                          self.results_with_preprocess['test_r2']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8, color='blue')
        bars2 = ax3.bar(x + width/2, enhanced_values, width, label='Enhanced', alpha=0.8, color='green')
        
        ax3.set_ylabel('Metric Values')
        ax3.set_title('Performance Metrics Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error Distribution (Baseline)
        ax4 = axes[1, 0]
        error_baseline = self.results_without_preprocess['y_pred_test'] - self.results_without_preprocess['y_test']
        ax4.hist(error_baseline, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax4.set_xlabel('Prediction Error')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Error Distribution (Baseline)')
        ax4.grid(True, alpha=0.3)
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8)
        
        # Plot 5: Error Distribution (Enhanced)
        ax5 = axes[1, 1]
        error_enhanced = self.results_with_preprocess['y_pred_test'] - self.results_with_preprocess['y_test']
        ax5.hist(error_enhanced, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax5.set_xlabel('Prediction Error')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Error Distribution (Enhanced)')
        ax5.grid(True, alpha=0.3)
        ax5.axvline(x=0, color='red', linestyle='--', alpha=0.8)
        
        # Plot 6: Feature Importance (Enhanced Model)
        ax6 = axes[1, 2]
        importance_dict = self.results_with_preprocess['feature_importance']
        top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        features, importances = zip(*top_features)
        
        ax6.barh(range(len(features)), importances, color='orange', alpha=0.7)
        ax6.set_yticks(range(len(features)))
        ax6.set_yticklabels([f.replace('_', ' ')[:15] for f in features])
        ax6.set_xlabel('Feature Importance')
        ax6.set_title('Top 10 Feature Importances (Enhanced)')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Save the figure
        try:
            plt.savefig('/content/turbofan_enhancement_results.png', dpi=300, bbox_inches='tight')
            print("📊 Visualization saved to: /content/turbofan_enhancement_results.png")
        except:
            print("📊 Visualization displayed successfully")

def run_turbofan_analysis(data_path=None, interactive_sensors=True):
    """Function to run the turbofan analysis with feature engineering comparison"""
    print("🚁 CMAPSS Turbofan Engine Degradation Analysis with Feature Engineering Enhancement")
    print("="*80)
    
    # Default path if none provided
    if data_path is None:
        data_path = "/content/extracted_data/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData"
    
    try:
        # Create analysis instance
        analysis = TurbofanAnalysis()
        
        # Load data
        print("📂 Loading CMAPSS datasets...")
        analysis.datasets = analysis.load_cmapss_data(data_path)
        
        if analysis.datasets:
            # Combine datasets
            print("\n🔄 Combining datasets...")
            analysis.combine_datasets()
            
            # Train models with comparison
            print("\n🤖 Training models for feature engineering comparison...")
            results_baseline, results_enhanced = analysis.train_models(interactive_selection=interactive_sensors)
            
            if results_baseline and results_enhanced:
                # Create comparison visualization
                analysis.create_comparison_visualization()
                print("\n✅ Analysis completed successfully!")
                return analysis
            else:
                print("❌ Model training failed!")
                return None
        else:
            print("❌ Failed to load datasets!")
            return None
            
    except Exception as e:
        print(f"❌ Error running analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Example usage function
def demo_analysis2():
    """Demo function to show how to use the analysis"""
    print("🚀 Running CMAPSS Turbofan Analysis Demo with Feature Engineering Enhancement")
    print("="*60)
    
    # Run analysis with interactive sensor selection
    analysis = run_turbofan_analysis(interactive_sensors=True)
    
    if analysis:
        print("\n📈 Analysis object created successfully!")
        print("Available features:")
        print("- Compare models: analysis.results_without_preprocess vs analysis.results_with_preprocess")
        print("- View recommended combinations: analysis.display_recommended_combinations()")
        print("- Access models: analysis.model_without_preprocessing, analysis.model_with_preprocessing")
        print("- Selected sensors: analysis.selected_sensors")
    
    return analysis

if __name__ == "__main__":
    # Run the demo
    demo_analysis2()
