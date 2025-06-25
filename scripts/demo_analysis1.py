import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
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
        self.model_without_normalization = None
        self.model_with_normalization = None
        self.results_without_norm = None
        self.results_with_norm = None

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

    def normalize_by_operating_conditions(self, data):
        """Improved normalization by operating conditions - fixed version"""
        normalized_data = data.copy()

        # Get operating condition columns and sensor columns
        condition_cols = ['setting1', 'setting2', 'setting3']
        sensor_cols = [f'sensor_{i}' for i in range(1, 22) if f'sensor_{i}' in data.columns]

        print(f"🔧 Normalizing {len(sensor_cols)} sensors by operating conditions...")

        # Check if operating conditions have meaningful variation
        operating_conditions = normalized_data[condition_cols].fillna(0)
        print(f"📊 Operating conditions stats:")
        for col in condition_cols:
            unique_vals = operating_conditions[col].nunique()
            std_val = operating_conditions[col].std()
            print(f"  {col}: {unique_vals} unique values, std = {std_val:.4f}")

        # If operating conditions have very little variation, use simpler normalization
        total_variance = operating_conditions.var().sum()
        print(f"📊 Total operating condition variance: {total_variance:.6f}")

        if total_variance < 1e-6:
            print("⚠️ Operating conditions have very low variance. Using global normalization instead.")
            # Apply global z-score normalization for each sensor
            for sensor in sensor_cols:
                mean_val = normalized_data[sensor].mean()
                std_val = normalized_data[sensor].std()
                if std_val > 0:
                    normalized_data[sensor] = (normalized_data[sensor] - mean_val) / std_val
            return normalized_data

        # Use operating condition-based normalization
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        # Standardize operating conditions first
        scaler = StandardScaler()
        operating_conditions_scaled = scaler.fit_transform(operating_conditions)

        # Determine optimal number of clusters
        n_samples = len(operating_conditions)
        min_samples_per_cluster = 50  # Ensure enough samples per cluster
        max_clusters = min(8, n_samples // min_samples_per_cluster)
        n_clusters = max(2, max_clusters)

        print(f"🔧 Using {n_clusters} clusters for operating condition normalization")

        # Create clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        normalized_data['operating_cluster'] = kmeans.fit_predict(operating_conditions_scaled)

        # Print cluster distribution
        cluster_counts = normalized_data['operating_cluster'].value_counts().sort_index()
        print(f"📊 Cluster distribution: {dict(cluster_counts)}")

        # Normalize sensors within each operating cluster
        for sensor in sensor_cols:
            sensor_normalized = []

            for cluster_id in range(n_clusters):
                cluster_mask = normalized_data['operating_cluster'] == cluster_id
                cluster_data = normalized_data.loc[cluster_mask, sensor]

                if len(cluster_data) > 1:
                    # Calculate cluster-specific statistics
                    cluster_mean = cluster_data.mean()
                    cluster_std = cluster_data.std()

                    # Avoid division by zero
                    if cluster_std > 1e-6:
                        cluster_normalized = (cluster_data - cluster_mean) / cluster_std
                    else:
                        cluster_normalized = cluster_data - cluster_mean
                else:
                    # If cluster has only one sample, just center it
                    cluster_normalized = cluster_data - cluster_data.mean()

                sensor_normalized.append(cluster_normalized)

            # Combine normalized values
            combined_normalized = pd.concat(sensor_normalized).reindex(normalized_data.index)
            normalized_data[sensor] = combined_normalized

        # Remove the cluster column
        normalized_data = normalized_data.drop('operating_cluster', axis=1)

        print("✅ Operating condition normalization completed")
        return normalized_data

    def prepare_modeling_data(self, selected_sensors, use_normalization=False):
        """Prepare data for modeling using selected sensors"""
        if use_normalization:
            print("🔄 Applying operating condition normalization...")
            train_data = self.normalize_by_operating_conditions(self.combined_train)
            test_data = self.normalize_by_operating_conditions(self.combined_test)
        else:
            print("📊 Using raw sensor data (no operating condition normalization)...")
            train_data = self.combined_train.copy()
            test_data = self.combined_test.copy()

        # Extract features and target
        X_train = train_data[selected_sensors].copy()
        y_train = train_data['RUL'].values
        X_test = test_data[selected_sensors].copy()
        y_test = test_data['RUL'].values

        # Handle missing values
        X_train = X_train.fillna(X_train.mean())
        X_test = X_test.fillna(X_train.mean())  # Use training mean for test data

        # Check for any remaining invalid values
        print(f"📊 Data quality check:")
        print(f"  Training data - NaN: {X_train.isnull().sum().sum()}, Inf: {np.isinf(X_train).sum().sum()}")
        print(f"  Test data - NaN: {X_test.isnull().sum().sum()}, Inf: {np.isinf(X_test).sum().sum()}")

        # Replace any infinite values
        X_train = X_train.replace([np.inf, -np.inf], 0)
        X_test = X_test.replace([np.inf, -np.inf], 0)

        # Standardize sensor features (only if not already normalized)
        if not use_normalization:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            # Convert to numpy arrays
            X_train = X_train.values
            X_test = X_test.values

        print(f"📈 Prepared data: Train shape={X_train.shape}, Test shape={X_test.shape}")

        return X_train, y_train, X_test, y_test

    def train_models(self, selected_sensors=None, interactive_selection=False):
        """Train models both with and without operating condition normalization"""
        if self.combined_train is None or self.combined_test is None:
            print("⚠️ Please load data first!")
            return

        try:
            # Get sensors based on selection method
            if interactive_selection:
                sensors = self.get_user_selected_sensors()
            elif selected_sensors is None:
                sensors = self.auto_select_sensors()
            else:
                sensors = selected_sensors

            self.selected_sensors = sensors

            print(f"\n🤖 Training models with {len(self.selected_sensors)} sensors...")
            print("Selected sensors for training:")
            for sensor in self.selected_sensors:
                print(f"  {sensor}: {self.sensor_descriptions[sensor]}")

            # Prepare data without normalization
            print(f"\n{'='*60}")
            print("📊 Training model WITHOUT operating condition normalization...")
            print(f"{'='*60}")
            X_train_no_norm, y_train, X_test_no_norm, y_test = self.prepare_modeling_data(sensors, use_normalization=False)

            # Train model without normalization
            self.model_without_normalization = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            self.model_without_normalization.fit(X_train_no_norm, y_train)

            # Make predictions without normalization
            y_pred_train_no_norm = self.model_without_normalization.predict(X_train_no_norm)
            y_pred_test_no_norm = self.model_without_normalization.predict(X_test_no_norm)

            # Calculate metrics without normalization
            self.results_without_norm = self._calculate_metrics(y_train, y_test, y_pred_train_no_norm, y_pred_test_no_norm)

            print(f"\n{'='*60}")
            print("📊 Training model WITH operating condition normalization...")
            print(f"{'='*60}")
            # Prepare data with normalization
            X_train_norm, _, X_test_norm, _ = self.prepare_modeling_data(sensors, use_normalization=True)

            # Train model with normalization
            self.model_with_normalization = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            self.model_with_normalization.fit(X_train_norm, y_train)

            # Make predictions with normalization
            y_pred_train_norm = self.model_with_normalization.predict(X_train_norm)
            y_pred_test_norm = self.model_with_normalization.predict(X_test_norm)

            # Calculate metrics with normalization
            self.results_with_norm = self._calculate_metrics(y_train, y_test, y_pred_train_norm, y_pred_test_norm)

            print("✅ Both models training completed!")
            self._print_comparison_results()
            self._analyze_normalization_effects()

            return self.results_without_norm, self.results_with_norm

        except Exception as e:
            print(f"❌ Error training models: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def _calculate_metrics(self, y_train, y_test, y_pred_train, y_pred_test):
        """Calculate performance metrics"""
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        return {
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }

    def _analyze_normalization_effects(self):
        """Analyze why normalization might be helping or hurting performance"""
        print(f"\n{'='*80}")
        print("🔍 ANALYSIS: Why Operating Condition Normalization Affects Performance")
        print(f"{'='*80}")

        # Check operating condition variations across datasets
        print("\n📊 Operating Condition Analysis:")
        condition_cols = ['setting1', 'setting2', 'setting3']

        for dataset in ['FD001', 'FD002', 'FD003', 'FD004']:
            if dataset in [d['train']['dataset'].iloc[0] for d in [{'train': self.combined_train[self.combined_train['dataset'] == dataset]}] if len(d['train']) > 0]:
                subset = self.combined_train[self.combined_train['dataset'] == dataset]
                print(f"\n  {dataset}:")
                for col in condition_cols:
                    unique_vals = subset[col].nunique()
                    std_val = subset[col].std()
                    min_val = subset[col].min()
                    max_val = subset[col].max()
                    print(f"    {col}: {unique_vals} unique values, range=[{min_val:.3f}, {max_val:.3f}], std={std_val:.4f}")

        # Analyze sensor correlations with operating conditions
        print(f"\n📈 Sensor-Operating Condition Correlations:")
        condition_sensor_corr = self.combined_train[condition_cols + self.selected_sensors].corr()

        strong_correlations = []
        for sensor in self.selected_sensors:
            for condition in condition_cols:
                corr_val = condition_sensor_corr.loc[sensor, condition]
                if abs(corr_val) > 0.3:  # Strong correlation threshold
                    strong_correlations.append((sensor, condition, corr_val))

        if strong_correlations:
            print("  Strong correlations found (|r| > 0.3):")
            for sensor, condition, corr in strong_correlations:
                print(f"    {sensor} ↔ {condition}: r = {corr:.3f}")
        else:
            print("  ⚠️ No strong correlations found between selected sensors and operating conditions")
            print("  This suggests operating condition normalization may not be beneficial for these sensors")

        # Analyze sensor variance before and after normalization
        print(f"\n📊 Sensor Variance Analysis:")

        # Calculate variances
        raw_data = self.combined_train[self.selected_sensors]
        normalized_data = self.normalize_by_operating_conditions(self.combined_train)[self.selected_sensors]

        print(f"  {'Sensor':<12} {'Raw Variance':<15} {'Norm Variance':<15} {'Variance Ratio':<15}")
        print("  " + "-" * 60)

        for sensor in self.selected_sensors:
            raw_var = raw_data[sensor].var()
            norm_var = normalized_data[sensor].var()
            ratio = norm_var / raw_var if raw_var > 0 else 0
            print(f"  {sensor:<12} {raw_var:<15.4f} {norm_var:<15.4f} {ratio:<15.4f}")

        # Performance difference analysis
        rmse_diff = self.results_with_norm['test_rmse'] - self.results_without_norm['test_rmse']
        mae_diff = self.results_with_norm['test_mae'] - self.results_without_norm['test_mae']
        r2_diff = self.results_with_norm['test_r2'] - self.results_without_norm['test_r2']

        print(f"\n🎯 Performance Impact Analysis:")
        print(f"  RMSE change: {rmse_diff:+.3f} ({'worse' if rmse_diff > 0 else 'better'})")
        print(f"  MAE change: {mae_diff:+.3f} ({'worse' if mae_diff > 0 else 'better'})")
        print(f"  R² change: {r2_diff:+.3f} ({'better' if r2_diff > 0 else 'worse'})")

        print(f"\n💡 Possible Explanations:")
        if len(strong_correlations) == 0:
            print("  1. ⚠️ Selected sensors may not be strongly affected by operating conditions")
            print("  2. 📊 Normalization may be removing useful variance patterns")
            print("  3. 🔧 The normalization approach may not be optimal for this sensor set")
        else:
            if rmse_diff > 0:  # Normalization made things worse
                print("  1. 📊 Normalization may be over-correcting, removing useful signal")
                print("  2. 🎯 The clustering approach may not capture true operating regimes")
                print("  3. ⚙️ Random Forest may already handle operating condition effects well")
            else:  # Normalization helped
                print("  1. ✅ Successfully removed operating condition bias")
                print("  2. 📈 Better generalization across different operating regimes")
                print("  3. 🎯 Cleaner signal-to-noise ratio for degradation detection")

        print(f"\n📋 Recommendations:")
        if rmse_diff > 5:  # Significant degradation
            print("  1. 🚫 Avoid operating condition normalization for this sensor combination")
            print("  2. 🔍 Consider feature engineering instead of normalization")
            print("  3. 📊 Try different normalization strategies (per-engine, per-flight-condition)")
        elif abs(rmse_diff) < 2:  # Small difference
            print("  1. 📊 Performance difference is minimal - either approach is acceptable")
            print("  2. 🎯 Consider computational cost and interpretability")
            print("  3. 🔧 May need more sophisticated normalization techniques")
        else:  # Normalization helped
            print("  1. ✅ Operating condition normalization is beneficial")
            print("  2. 📈 Consider this approach for production models")
            print("  3. 🔧 Fine-tune normalization parameters for better results")

    def _print_comparison_results(self):
        """Print comparison of results between normalized and non-normalized models"""
        print("\n" + "="*80)
        print("📊 MODEL COMPARISON RESULTS")
        print("="*80)

        print(f"\n🔧 Selected Sensors: {len(self.selected_sensors)}")
        for sensor in self.selected_sensors:
            print(f"   {sensor}: {self.sensor_descriptions[sensor]}")

        print(f"\n📈 Performance Comparison:")
        print(f"{'Metric':<15} {'Without Norm':<15} {'With Norm':<15} {'Improvement':<15}")
        print("-" * 60)

        # Calculate improvements correctly
        rmse_improvement = ((self.results_without_norm['test_rmse'] - self.results_with_norm['test_rmse']) / self.results_without_norm['test_rmse']) * 100
        mae_improvement = ((self.results_without_norm['test_mae'] - self.results_with_norm['test_mae']) / self.results_without_norm['test_mae']) * 100
        r2_improvement = ((self.results_with_norm['test_r2'] - self.results_without_norm['test_r2']) / abs(self.results_without_norm['test_r2'])) * 100

        metrics = [
            ('Test RMSE', self.results_without_norm['test_rmse'], self.results_with_norm['test_rmse'], rmse_improvement),
            ('Test MAE', self.results_without_norm['test_mae'], self.results_with_norm['test_mae'], mae_improvement),
            ('Test R²', self.results_without_norm['test_r2'], self.results_with_norm['test_r2'], r2_improvement)
        ]

        for metric_name, without_norm, with_norm, improvement in metrics:
            if improvement > 0:
                improvement_str = f"✅ {improvement:.2f}%"
            else:
                improvement_str = f"❌ {improvement:.2f}%"

            print(f"{metric_name:<15} {without_norm:<15.3f} {with_norm:<15.3f} {improvement_str:<15}")

        # Summary
        print(f"\n📋 Summary:")
        if rmse_improvement > 0 and mae_improvement > 0:
            print("🎉 Operating condition normalization IMPROVED model performance!")
        elif rmse_improvement < 0 and mae_improvement < 0:
            print("⚠️ Operating condition normalization DECREASED model performance")
        else:
            print("📊 Operating condition normalization shows mixed results")

    def create_comparison_visualization(self):
        """Create comprehensive comparison visualization between normalized and non-normalized models"""
        if self.results_without_norm is None or self.results_with_norm is None:
            print("⚠️ No results to visualize!")
            return

        # Force matplotlib to use inline backend for Colab
        try:
            from IPython import get_ipython
            get_ipython().run_line_magic('matplotlib', 'inline')
        except:
            pass

        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Model Comparison Analysis - Operating Condition Normalization vs Non-Normalization\nUsing {len(self.selected_sensors)} Sensors',
                     fontsize=16, fontweight='bold')

        # Sample data for visualization (to avoid overcrowding)
        sample_size = min(2000, len(self.results_without_norm['y_test']))
        sample_indices = np.random.choice(len(self.results_without_norm['y_test']), sample_size, replace=False)

        # Plot 1: Prediction vs True (Without Normalization)
        ax1 = axes[0, 0]
        y_test = self.results_without_norm['y_test']
        y_pred_test = self.results_without_norm['y_pred_test']
        ax1.scatter(y_test[sample_indices], y_pred_test[sample_indices], alpha=0.6, s=15, color='blue', label='Predicted Values')
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction Line')
        ax1.set_xlabel('True RUL Values')
        ax1.set_ylabel('Predicted RUL Values')
        ax1.set_title('Non-Normalized Model Prediction Results')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.text(0.05, 0.95, f'R² = {self.results_without_norm["test_r2"]:.3f}\nRMSE = {self.results_without_norm["test_rmse"]:.2f}',
                transform=ax1.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

        # Plot 2: Prediction vs True (With Normalization)
        ax2 = axes[0, 1]
        y_test_norm = self.results_with_norm['y_test']
        y_pred_test_norm = self.results_with_norm['y_pred_test']
        ax2.scatter(y_test_norm[sample_indices], y_pred_test_norm[sample_indices], alpha=0.6, s=15, color='green', label='Predicted Values')
        ax2.plot([y_test_norm.min(), y_test_norm.max()], [y_test_norm.min(), y_test_norm.max()], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction Line')
        ax2.set_xlabel('True RUL Values')
        ax2.set_ylabel('Predicted RUL Values')
        ax2.set_title('Normalized Model Prediction Results')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.text(0.05, 0.95, f'R² = {self.results_with_norm["test_r2"]:.3f}\nRMSE = {self.results_with_norm["test_rmse"]:.2f}',
                transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

        # Plot 3: Metrics Comparison Bar Chart
        ax3 = axes[0, 2]
        metrics = ['RMSE', 'MAE', 'R²']
        without_norm_values = [self.results_without_norm['test_rmse'],
                              self.results_without_norm['test_mae'],
                              self.results_without_norm['test_r2']]
        with_norm_values = [self.results_with_norm['test_rmse'],
                           self.results_with_norm['test_mae'],
                           self.results_with_norm['test_r2']]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax3.bar(x - width/2, without_norm_values, width, label='Non-Normalized', alpha=0.8, color='blue')
        bars2 = ax3.bar(x + width/2, with_norm_values, width, label='Normalized', alpha=0.8, color='green')

        # Add value labels on bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            ax3.text(bar1.get_x() + bar1.get_width()/2., height1 + max(without_norm_values + with_norm_values) * 0.01,
                    f'{height1:.3f}', ha='center', va='bottom', fontsize=9)
            ax3.text(bar2.get_x() + bar2.get_width()/2., height2 + max(without_norm_values + with_norm_values) * 0.01,
                    f'{height2:.3f}', ha='center', va='bottom', fontsize=9)

        ax3.set_ylabel('Metric Values')
        ax3.set_title('Performance Metrics Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Error Distribution (Without Normalization)
        ax4 = axes[1, 0]
        error_without = self.results_without_norm['y_pred_test'] - self.results_without_norm['y_test']
        ax4.hist(error_without, bins=50, alpha=0.7, density=True, color='blue', edgecolor='black')
        ax4.set_xlabel('Prediction Error (Predicted - True)')
        ax4.set_ylabel('Density')
        ax4.set_title('Error Distribution (Non-Normalized)')
        ax4.grid(True, alpha=0.3)
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax4.axvline(x=np.mean(error_without), color='orange', linestyle='-', alpha=0.8, linewidth=2, label=f'Mean: {np.mean(error_without):.2f}')
        ax4.legend()
        ax4.text(0.05, 0.95, f'Std Dev = {np.std(error_without):.2f}\nSkewness = {pd.Series(error_without).skew():.3f}',
                transform=ax4.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

        # Plot 5: Error Distribution (With Normalization)
        ax5 = axes[1, 1]
        error_with = self.results_with_norm['y_pred_test'] - self.results_with_norm['y_test']
        ax5.hist(error_with, bins=50, alpha=0.7, density=True, color='green', edgecolor='black')
        ax5.set_xlabel('Prediction Error (Predicted - True)')
        ax5.set_ylabel('Density')
        ax5.set_title('Error Distribution (Normalized)')
        ax5.grid(True, alpha=0.3)
        ax5.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax5.axvline(x=np.mean(error_with), color='orange', linestyle='-', alpha=0.8, linewidth=2, label=f'Mean: {np.mean(error_with):.2f}')
        ax5.legend()
        ax5.text(0.05, 0.95, f'Std Dev = {np.std(error_with):.2f}\nSkewness = {pd.Series(error_with).skew():.3f}',
                transform=ax5.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

        # Plot 6: Improvement percentage
        ax6 = axes[1, 2]
        rmse_improvement = ((self.results_without_norm['test_rmse'] - self.results_with_norm['test_rmse']) / self.results_without_norm['test_rmse']) * 100
        mae_improvement = ((self.results_without_norm['test_mae'] - self.results_with_norm['test_mae']) / self.results_without_norm['test_mae']) * 100
        r2_improvement = ((self.results_with_norm['test_r2'] - self.results_without_norm['test_r2']) / abs(self.results_without_norm['test_r2'])) * 100

        improvements = [rmse_improvement, mae_improvement, r2_improvement]
        colors = ['green' if x > 0 else 'red' for x in improvements]

        bars = ax6.bar(metrics, improvements, color=colors, alpha=0.7, edgecolor='black')

        # Add value labels
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                    f'{improvement:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')

        ax6.set_ylabel('Improvement Percentage (%)')
        ax6.set_title('Normalized vs Non-Normalized Improvement')
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        plt.tight_layout()

        # Display in Colab
        plt.show()

        # Also save the figure
        try:
            plt.savefig('/content/turbofan_comparison_results.png', dpi=300, bbox_inches='tight')
            print("📊 Comparison visualization saved to: /content/turbofan_comparison_results.png")
        except:
            print("📊 Comparison visualization displayed successfully")

def run_turbofan_analysis(data_path=None, interactive_sensors=True):
    """Function to run the turbofan analysis with operating condition normalization comparison"""
    print("🚁 CMAPSS Turbofan Engine Degradation Analysis with Operating Condition Normalization Comparison")
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
            print("\n🤖 Training models for operating condition normalization comparison...")
            results_no_norm, results_with_norm = analysis.train_models(interactive_selection=interactive_sensors)

            if results_no_norm and results_with_norm:
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
def demo_analysis1():
    """Demo function to show how to use the analysis"""
    print("🚀 Running CMAPSS Turbofan Analysis Demo with Operating Condition Normalization Comparison")
    print("="*60)

    # Run analysis with interactive sensor selection
    analysis = run_turbofan_analysis(interactive_sensors=True)

    if analysis:
        print("\n📈 Analysis object created successfully!")
        print("Available features:")
        print("- Compare models: analysis.results_without_norm vs analysis.results_with_norm")
        print("- View recommended combinations: analysis.display_recommended_combinations()")
        print("- Access models: analysis.model_without_normalization, analysis.model_with_normalization")
        print("- Selected sensors: analysis.selected_sensors")
        print("- Re-run with different sensors: analysis.train_models(['sensor_1', 'sensor_2', ...])")

    return analysis

if __name__ == "__main__":
    # Run the demo
    demo_analysis1()
