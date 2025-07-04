def comprehensive_model_evaluation_and_visualization(model， test_loader, save_path='best_enhanced_model.pth',
                                                   sensor_channels=list(range(1, 22)),
                                                   n_engines=6, figsize=(16, 20)):
    """
    Comprehensive evaluation and visualization function for the enhanced LSTM autoencoder model.
    
    Args:
        model: The trained model to evaluate
        test_loader: DataLoader for test data
        save_path: Path to the saved model checkpoint
        sensor_channels: List of sensor channel indices to visualize
        n_engines: Number of engines to show in multi-engine plot
        figsize: Figure size for the plots
    
    Returns:
        dict: Dictionary containing evaluation metrics and results
    """
    
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load best model
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Set style for publication-quality plots
    plt.style.use('default')
    sns.set_palette("deep")

    # Collect predictions and ground truth - store each engine separately
    engine_data = []

    print("Collecting model predictions...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            x_full = batch['x'].to(device)
            rul_full = batch['rul'].to(device)
            mask = batch['mask'].to(device)

            x_in, x_tgt = x_full[:, :-1], x_full[:, 1:]
            rul_tgt, m = rul_full[:, 1:], mask[:, 1:]

            x_pred, rul_pred = model(x_in)

            # Store results for each engine in the batch
            for i in range(x_full.size(0)):
                engine_data.append({
                    'prediction': rul_pred[i].cpu().numpy(),
                    'target': rul_tgt[i].cpu().numpy(),
                    'reconstruction': x_pred[i].cpu().numpy(),
                    'original': x_tgt[i].cpu().numpy(),
                    'mask': m[i].cpu().numpy(),
                    'engine_id': batch_idx * test_loader.batch_size + i
                })

    print(f"Collected data for {len(engine_data)} engines")

    # Calculate comprehensive evaluation metrics
    def calculate_comprehensive_metrics(predictions, targets):
        """
        Calculate RMSE, Score (Time-Deviation Penalty Index), and Accuracy metrics
        
        Args:
            predictions: predicted RUL values
            targets: ground truth RUL values
            
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        # Calculate deviation
        dm = predictions - targets
        M = len(dm)
        
        # (1) Root Mean Square Error (RMSE)
        rmse = np.sqrt(np.mean(dm ** 2))
        
        # (2) Score (Time-Deviation Penalty Index)
        # Apply asymmetric penalties: early predictions (dm < 0) and late predictions (dm >= 0)
        score_values = np.where(dm < 0, 
                               np.exp(-dm / 13) - 1,  # Early prediction penalty
                               np.exp(dm / 10) - 1)   # Late prediction penalty (higher)
        score = np.sum(score_values)
        
        # (3) Accuracy (Tolerance-Based)
        # Percentage of predictions within acceptable error range [-13, +10] cycles
        tolerance_mask = (dm >= -13) & (dm <= 10)
        accuracy = np.mean(tolerance_mask) * 100
        
        return {
            'rmse': rmse,
            'score': score,
            'accuracy': accuracy,
            'deviations': dm,
            'n_samples': M
        }

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 0.8, 1.2], hspace=0.4, wspace=0.25,
                         left=0.08, right=0.95, top=0.95, bottom=0.05)

    # Color scheme
    colors = {
        'predicted': '#2E86AB',  # Blue
        'actual': '#A23B72',     # Magenta
        'reconstructed': '#F18F01',  # Orange
        'original': '#C73E1D'    # Red
    }

    # Select engines for visualization
    n_total_engines = len(engine_data)
    selected_engines = np.linspace(0, n_total_engines-1, min(n_engines, n_total_engines), dtype=int)

    # Plot 1: RUL Prediction Comparison (Multiple Engines)
    ax1 = fig.add_subplot(gs[0, :])

    for i, engine_idx in enumerate(selected_engines):
        engine = engine_data[engine_idx]
        pred = engine['prediction']
        target = engine['target']
        mask = engine['mask']

        # Find valid indices
        valid_idx = mask.astype(bool)
        if not np.any(valid_idx):
            continue

        time_steps = np.arange(len(pred))[valid_idx]
        pred_valid = pred[valid_idx]
        target_valid = target[valid_idx]

        alpha = 0.8 if i == 0 else 0.6
        linewidth = 2.5 if i == 0 else 1.8

        if i == 0:
            ax1.plot(time_steps, target_valid, color=colors['actual'],
                    linewidth=linewidth, alpha=alpha, label='Ground Truth')
            ax1.plot(time_steps, pred_valid, color=colors['predicted'],
                    linewidth=linewidth, alpha=alpha, label='Predicted')
        else:
            ax1.plot(time_steps, target_valid, color=colors['actual'],
                    linewidth=linewidth, alpha=alpha)
            ax1.plot(time_steps, pred_valid, color=colors['predicted'],
                    linewidth=linewidth, alpha=alpha)

    ax1.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Remaining Useful Life (RUL)', fontsize=12, fontweight='bold')
    ax1.set_title('RUL Prediction Performance Across Multiple Engines',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Plot 2: Detailed RUL for Single Engine
    ax2 = fig.add_subplot(gs[1, :])

    # Select engine with longest valid sequence for detailed view
    engine_lengths = [np.sum(engine['mask'].astype(bool)) for engine in engine_data]
    longest_engine_idx = np.argmax(engine_lengths)
    longest_engine = engine_data[longest_engine_idx]

    pred_detailed = longest_engine['prediction']
    target_detailed = longest_engine['target']
    mask_detailed = longest_engine['mask']
    valid_detailed = mask_detailed.astype(bool)

    time_detailed = np.arange(len(pred_detailed))[valid_detailed]
    pred_valid_detailed = pred_detailed[valid_detailed]
    target_valid_detailed = target_detailed[valid_detailed]

    # Calculate prediction intervals (confidence bands)
    residuals = pred_valid_detailed - target_valid_detailed
    std_residual = np.std(residuals)

    ax2.fill_between(time_detailed,
                     pred_valid_detailed - 1.96 * std_residual,
                     pred_valid_detailed + 1.96 * std_residual,
                     alpha=0.2, color=colors['predicted'], label='95% Confidence')

    ax2.plot(time_detailed, target_valid_detailed, color=colors['actual'],
             linewidth=3, label='Ground Truth', marker='o', markersize=4, alpha=0.8)
    ax2.plot(time_detailed, pred_valid_detailed, color=colors['predicted'],
             linewidth=3, label='Predicted', marker='s', markersize=3, alpha=0.9)

    # Calculate comprehensive metrics for this engine
    detailed_metrics = calculate_comprehensive_metrics(pred_valid_detailed, target_valid_detailed)

    ax2.text(0.02, 0.98, 
             f'RMSE: {detailed_metrics["rmse"]:.3f}\n'
             f'Score: {detailed_metrics["score"]:.3f}\n'
             f'Accuracy: {detailed_metrics["accuracy"]:.1f}%',
             transform=ax2.transAxes, fontsize=11, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Remaining Useful Life (RUL)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Detailed RUL Prediction for Engine {longest_engine["engine_id"] + 1}',
                  fontsize=14, fontweight='bold', pad=20)
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Plot 3: Evaluation Metrics Visualization
    ax3_left = fig.add_subplot(gs[2, 0])
    ax3_center = fig.add_subplot(gs[2, 1])
    ax3_right = fig.add_subplot(gs[2, 2])

    # Calculate overall metrics
    all_valid_predictions = []
    all_valid_targets = []

    for engine in engine_data:
        mask = engine['mask'].astype(bool)
        if np.any(mask):
            all_valid_predictions.extend(engine['prediction'][mask])
            all_valid_targets.extend(engine['target'][mask])

    all_valid_predictions = np.array(all_valid_predictions)
    all_valid_targets = np.array(all_valid_targets)
    overall_metrics = calculate_comprehensive_metrics(all_valid_predictions, all_valid_targets)

    # Error distribution histogram
    ax3_left.hist(overall_metrics['deviations'], bins=50, alpha=0.7, color=colors['predicted'], edgecolor='black')
    ax3_left.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    ax3_left.axvline(-13, color='orange', linestyle='--', alpha=0.7, label='Tolerance Range')
    ax3_left.axvline(10, color='orange', linestyle='--', alpha=0.7)
    ax3_left.set_xlabel('Prediction Error (cycles)', fontsize=10, fontweight='bold')
    ax3_left.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax3_left.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax3_left.legend(fontsize=9)
    ax3_left.grid(True, alpha=0.3)

    # Metrics bar plot
    metric_names = ['RMSE', 'Score', 'Accuracy (%)']
    metric_values = [overall_metrics['rmse'], overall_metrics['score'], overall_metrics['accuracy']]
    bars = ax3_center.bar(metric_names, metric_values, color=[colors['predicted'], colors['actual'], colors['reconstructed']])
    ax3_center.set_title('Performance Metrics', fontsize=12, fontweight='bold')
    ax3_center.set_ylabel('Metric Value', fontsize=10, fontweight='bold')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax3_center.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

    # Scatter plot: Predicted vs Actual
    ax3_right.scatter(all_valid_targets, all_valid_predictions, alpha=0.6, color=colors['predicted'], s=20)
    min_rul = min(np.min(all_valid_targets), np.min(all_valid_predictions))
    max_rul = max(np.max(all_valid_targets), np.max(all_valid_predictions))
    ax3_right.plot([min_rul, max_rul], [min_rul, max_rul], 'r--', linewidth=2, label='Perfect Prediction')
    ax3_right.set_xlabel('Actual RUL', fontsize=10, fontweight='bold')
    ax3_right.set_ylabel('Predicted RUL', fontsize=10, fontweight='bold')
    ax3_right.set_title('Predicted vs Actual RUL', fontsize=12, fontweight='bold')
    ax3_right.legend(fontsize=9)
    ax3_right.grid(True, alpha=0.3)

    # Plot 4: Sensor Reconstruction for Selected Channels
    n_sensors = len(sensor_channels)
    n_cols = 3
    n_rows = (n_sensors + n_cols - 1) // n_cols

    gs_sensors = gs[3, :].subgridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)

    # Use the same engine as detailed RUL plot
    orig_sensors = longest_engine['original']  # [seq_len, features]
    recon_sensors = longest_engine['reconstruction']  # [seq_len, features]

    for i, sensor_idx in enumerate(sensor_channels):
        if i >= n_rows * n_cols:
            break

        row = i // n_cols
        col = i % n_cols
        ax_sensor = fig.add_subplot(gs_sensors[row, col])

        orig_signal = orig_sensors[valid_detailed, sensor_idx]
        recon_signal = recon_sensors[valid_detailed, sensor_idx]

        ax_sensor.plot(time_detailed, orig_signal, color=colors['original'],
                      linewidth=2, alpha=0.8, label='Original')
        ax_sensor.plot(time_detailed, recon_signal, color=colors['reconstructed'],
                      linewidth=2, alpha=0.8, label='Reconstructed')

        # Calculate reconstruction error
        recon_mse = np.mean((orig_signal - recon_signal) ** 2)

        ax_sensor.set_title(f'Sensor {sensor_idx + 1}\nMSE: {recon_mse:.4f}',
                           fontsize=10, fontweight='bold')
        ax_sensor.set_xlabel('Time Steps', fontsize=9)
        ax_sensor.set_ylabel('Sensor Value', fontsize=9)
        ax_sensor.grid(True, alpha=0.3, linestyle='--')
        ax_sensor.spines['top'].set_visible(False)
        ax_sensor.spines['right'].set_visible(False)

        if i == 0:
            ax_sensor.legend(loc='best', fontsize=8)

    # Remove empty subplots
    for i in range(len(sensor_channels), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(fig.add_subplot(gs_sensors[row, col]))

    # Add main title
    fig.suptitle('Enhanced LSTM Autoencoder for Turbofan Engine RUL Prediction',
                 fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    plt.savefig('enhanced_rul_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.savefig('enhanced_rul_prediction_results.pdf', bbox_inches='tight')

    print("Comprehensive evaluation plots saved successfully!")
    print("Files: enhanced_rul_prediction_results.png, enhanced_rul_prediction_results.pdf")

    plt.show()

    # Print comprehensive evaluation summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE MODEL EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Overall Test RMSE:      {overall_metrics['rmse']:.6f}")
    print(f"Overall Test Score:      {overall_metrics['score']:.6f}")
    print(f"Overall Test Accuracy:   {overall_metrics['accuracy']:.2f}%")
    print(f"Total Test Samples:      {overall_metrics['n_samples']:,}")
    print(f"Number of Engines:       {len(engine_data)}")
    print(f"\nMETRIC EXPLANATIONS:")
    print(f"- RMSE: Root Mean Square Error (lower is better)")
    print(f"- Score: Time-Deviation Penalty Index with asymmetric penalties (lower is better)")
    print(f"- Accuracy: Percentage of predictions within tolerance [-13, +10] cycles (higher is better)")
    print(f"{'='*80}")

    return {
        'rmse': overall_metrics['rmse'],
        'score': overall_metrics['score'],
        'accuracy': overall_metrics['accuracy'],
        'n_samples': overall_metrics['n_samples'],
        'n_engines': len(engine_data),
        'engine_data': engine_data,
        'detailed_engine_metrics': {
            'rmse': detailed_metrics['rmse'],
            'score': detailed_metrics['score'],
            'accuracy': detailed_metrics['accuracy'],
            'engine_id': longest_engine['engine_id']
        },
        'all_metrics': overall_metrics
    }


# 调用封装后的函数
plot_results = comprehensive_model_evaluation_and_visualization(
    model=trained_model,
    test_loader=test_loader,
    save_path='best_enhanced_model.pth',
    sensor_channels=[1, 2, 3, 4, 11, 12, 13, 15, 17, 20],
    n_engines=6,
    figsize=(16, 20)
)

print(f"\nVisualization completed! Overall test performance:")
print(f"RMSE: {plot_results['rmse']:.6f}")
print(f"Score: {plot_results['score']:.6f}")
print(f"Accuracy: {plot_results['accuracy']:.2f}%")
