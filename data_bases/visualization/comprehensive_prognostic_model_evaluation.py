# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm

# def comprehensive_model_evaluation(model, model_save_path, test_loader, 
#                                  sensor_channels=list(range(1, 22)),
#                                  n_engines=6, figsize=(16, 20)):
#     """
#     Comprehensive evaluation and visualization of the enhanced LSTM autoencoder model.
    
#     Args:
#         model: The model instance to evaluate
#         model_save_path (str): Path to the saved model checkpoint
#         test_loader: DataLoader for test data
#         sensor_channels (list): List of sensor channels to visualize
#         n_engines (int): Number of engines to display in multi-engine plot
#         figsize (tuple): Figure size for the plots
    
#     Returns:
#         dict: Dictionary containing all evaluation metrics
#     """
    
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     # Load best model - fix the loading issue
#     checkpoint = torch.load(model_save_path, map_location=device)
    
#     # Handle different checkpoint formats
#     if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#         model.load_state_dict(checkpoint['model_state_dict'])
#     else:
#         # If the checkpoint is just the state_dict directly (as saved by torch.save(model.state_dict(), path))
#         model.load_state_dict(checkpoint)
    
#     model.to(device)
#     model.eval()

#     # Set style for publication-quality plots
#     plt.style.use('default')
#     sns.set_palette("deep")

#     # Collect predictions and ground truth - store each engine separately
#     engine_data = []

#     print("Collecting model predictions...")
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(tqdm(test_loader)):
#             x_full = batch['x'].to(device)
#             rul_full = batch['rul'].to(device)
#             mask = batch['mask'].to(device)

#             x_in, x_tgt = x_full[:, :-1], x_full[:, 1:]
#             rul_tgt, m = rul_full[:, 1:], mask[:, 1:]

#             x_pred, rul_pred = model(x_in)

#             # Store results for each engine in the batch
#             for i in range(x_full.size(0)):
#                 engine_data.append({
#                     'prediction': rul_pred[i].cpu().numpy(),
#                     'target': rul_tgt[i].cpu().numpy(),
#                     'reconstruction': x_pred[i].cpu().numpy(),
#                     'original': x_tgt[i].cpu().numpy(),
#                     'mask': m[i].cpu().numpy(),
#                     'engine_id': batch_idx * test_loader.batch_size + i
#                 })

#     print(f"Collected data for {len(engine_data)} engines")

#     # Create figure with subplots
#     fig = plt.figure(figsize=figsize)
#     gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.25,
#                          left=0.08, right=0.95, top=0.95, bottom=0.05)

#     # Color scheme
#     colors = {
#         'predicted': '#2E86AB',  # Blue
#         'actual': '#A23B72',     # Magenta
#         'reconstructed': '#F18F01',  # Orange
#         'original': '#C73E1D'    # Red
#     }

#     # Select engines for visualization
#     n_total_engines = len(engine_data)
#     selected_engines = np.linspace(0, n_total_engines-1, min(n_engines, n_total_engines), dtype=int)

#     # Plot 1: RUL Prediction Comparison (Multiple Engines)
#     ax1 = fig.add_subplot(gs[0, :])

#     for i, engine_idx in enumerate(selected_engines):
#         engine = engine_data[engine_idx]
#         pred = engine['prediction']
#         target = engine['target']
#         mask = engine['mask']

#         # Find valid indices
#         valid_idx = mask.astype(bool)
#         if not np.any(valid_idx):
#             continue

#         time_steps = np.arange(len(pred))[valid_idx]
#         pred_valid = pred[valid_idx]
#         target_valid = target[valid_idx]

#         alpha = 0.8 if i == 0 else 0.6
#         linewidth = 2.5 if i == 0 else 1.8

#         if i == 0:
#             ax1.plot(time_steps, target_valid, color=colors['actual'],
#                     linewidth=linewidth, alpha=alpha, label='Ground Truth')
#             ax1.plot(time_steps, pred_valid, color=colors['predicted'],
#                     linewidth=linewidth, alpha=alpha, label='Predicted')
#         else:
#             ax1.plot(time_steps, target_valid, color=colors['actual'],
#                     linewidth=linewidth, alpha=alpha)
#             ax1.plot(time_steps, pred_valid, color=colors['predicted'],
#                     linewidth=linewidth, alpha=alpha)

#     ax1.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
#     ax1.set_ylabel('Remaining Useful Life (RUL)', fontsize=12, fontweight='bold')
#     ax1.set_title('RUL Prediction Performance Across Multiple Engines',
#                   fontsize=14, fontweight='bold', pad=20)
#     ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
#     ax1.grid(True, alpha=0.3, linestyle='--')
#     ax1.spines['top'].set_visible(False)
#     ax1.spines['right'].set_visible(False)

#     # Plot 2: Detailed RUL for Single Engine
#     ax2 = fig.add_subplot(gs[1, :])

#     # Select engine with longest valid sequence for detailed view
#     engine_lengths = [np.sum(engine['mask'].astype(bool)) for engine in engine_data]
#     longest_engine_idx = np.argmax(engine_lengths)
#     longest_engine = engine_data[longest_engine_idx]

#     pred_detailed = longest_engine['prediction']
#     target_detailed = longest_engine['target']
#     mask_detailed = longest_engine['mask']
#     valid_detailed = mask_detailed.astype(bool)

#     time_detailed = np.arange(len(pred_detailed))[valid_detailed]
#     pred_valid_detailed = pred_detailed[valid_detailed]
#     target_valid_detailed = target_detailed[valid_detailed]

#     # Calculate prediction intervals (confidence bands)
#     residuals = pred_valid_detailed - target_valid_detailed
#     std_residual = np.std(residuals)

#     ax2.fill_between(time_detailed,
#                      pred_valid_detailed - 1.96 * std_residual,
#                      pred_valid_detailed + 1.96 * std_residual,
#                      alpha=0.2, color=colors['predicted'], label='95% Confidence')

#     ax2.plot(time_detailed, target_valid_detailed, color=colors['actual'],
#              linewidth=3, label='Ground Truth', marker='o', markersize=4, alpha=0.8)
#     ax2.plot(time_detailed, pred_valid_detailed, color=colors['predicted'],
#              linewidth=3, label='Predicted', marker='s', markersize=3, alpha=0.9)

#     # Calculate and display metrics
#     mse = np.mean((pred_valid_detailed - target_valid_detailed) ** 2)
#     mae = np.mean(np.abs(pred_valid_detailed - target_valid_detailed))

#     ax2.text(0.02, 0.98, f'MSE: {mse:.3f}\nMAE: {mae:.3f}',
#              transform=ax2.transAxes, fontsize=11, fontweight='bold',
#              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

#     ax2.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
#     ax2.set_ylabel('Remaining Useful Life (RUL)', fontsize=12, fontweight='bold')
#     ax2.set_title(f'Detailed RUL Prediction for Engine {longest_engine["engine_id"] + 1}',
#                   fontsize=14, fontweight='bold', pad=20)
#     ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
#     ax2.grid(True, alpha=0.3, linestyle='--')
#     ax2.spines['top'].set_visible(False)
#     ax2.spines['right'].set_visible(False)

#     # Plot 3: Sensor Reconstruction for Selected Channels
#     n_sensors = len(sensor_channels)
#     n_cols = 3
#     n_rows = (n_sensors + n_cols - 1) // n_cols

#     gs_sensors = gs[2, :].subgridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)

#     # Use the same engine as detailed RUL plot
#     orig_sensors = longest_engine['original']  # [seq_len, features]
#     recon_sensors = longest_engine['reconstruction']  # [seq_len, features]

#     for i, sensor_idx in enumerate(sensor_channels):
#         if i >= n_rows * n_cols:
#             break

#         row = i // n_cols
#         col = i % n_cols
#         ax_sensor = fig.add_subplot(gs_sensors[row, col])

#         orig_signal = orig_sensors[valid_detailed, sensor_idx]
#         recon_signal = recon_sensors[valid_detailed, sensor_idx]

#         ax_sensor.plot(time_detailed, orig_signal, color=colors['original'],
#                       linewidth=2, alpha=0.8, label='Original')
#         ax_sensor.plot(time_detailed, recon_signal, color=colors['reconstructed'],
#                       linewidth=2, alpha=0.8, label='Reconstructed')

#         # Calculate reconstruction error
#         recon_mse = np.mean((orig_signal - recon_signal) ** 2)

#         ax_sensor.set_title(f'Sensor {sensor_idx + 1}\nMSE: {recon_mse:.4f}',
#                            fontsize=10, fontweight='bold')
#         ax_sensor.set_xlabel('Time Steps', fontsize=9)
#         ax_sensor.set_ylabel('Sensor Value', fontsize=9)
#         ax_sensor.grid(True, alpha=0.3, linestyle='--')
#         ax_sensor.spines['top'].set_visible(False)
#         ax_sensor.spines['right'].set_visible(False)

#         if i == 0:
#             ax_sensor.legend(loc='best', fontsize=8)

#     # Remove empty subplots
#     for i in range(len(sensor_channels), n_rows * n_cols):
#         row = i // n_cols
#         col = i % n_cols
#         fig.delaxes(fig.add_subplot(gs_sensors[row, col]))

#     # Add main title
#     fig.suptitle('Enhanced LSTM Autoencoder for Turbofan Engine RUL Prediction',
#                  fontsize=16, fontweight='bold', y=0.98)

#     # Save figure
#     plt.savefig('enhanced_rul_prediction_results.png', dpi=300, bbox_inches='tight')
#     plt.savefig('enhanced_rul_prediction_results.pdf', bbox_inches='tight')

#     print("Files: enhanced_rul_prediction_results.png, enhanced_rul_prediction_results.pdf")

#     plt.show()

#     # Calculate comprehensive evaluation metrics
#     print(f"\n{'='*80}")
#     print("COMPREHENSIVE MODEL EVALUATION METRICS")
#     print(f"{'='*80}")

#     # Collect all valid predictions and targets
#     all_valid_predictions = []
#     all_valid_targets = []

#     for engine in engine_data:
#         mask = engine['mask'].astype(bool)
#         if np.any(mask):
#             all_valid_predictions.extend(engine['prediction'][mask])
#             all_valid_targets.extend(engine['target'][mask])

#     all_valid_predictions = np.array(all_valid_predictions)
#     all_valid_targets = np.array(all_valid_targets)

#     # Calculate deviation (d_m = predicted - actual)
#     deviations = all_valid_predictions - all_valid_targets

#     # (1) Root Mean Square Error (RMSE)
#     mse = np.mean(deviations ** 2)
#     rmse = np.sqrt(mse)

#     # (2) Score (Time-Deviation Penalty Index)
#     score_values = np.where(
#         deviations < 0,
#         np.exp(-deviations / 13) - 1,  # Early predictions (less penalty)
#         np.exp(deviations / 10) - 1    # Late predictions (more penalty)
#     )
#     score = np.sum(score_values)

#     # (3) Accuracy (Tolerance-Based)
#     tolerance_mask = (deviations >= -13) & (deviations <= 10)
#     accuracy = np.mean(tolerance_mask) * 100

#     # Additional metrics
#     mae = np.mean(np.abs(deviations))

#     # Print results
#     print(f"Root Mean Square Error (RMSE): {rmse:.6f}")
#     print(f"Score (Time-Deviation Penalty): {score:.6f}")
#     print(f"Accuracy (Tolerance [-13, +10]): {accuracy:.2f}%")

#     print(f"{'='*80}")

#     return {
#         'rmse': rmse,
#         'score': score,
#         'accuracy': accuracy,
#         'mae': mae,
#         'mse': mse,
#         'mean_deviation': np.mean(deviations),
#         'std_deviation': np.std(deviations),
#         'n_samples': len(all_valid_predictions),
#         'n_engines': len(engine_data),
#         'late_predictions_pct': np.mean(deviations > 0) * 100,
#         'early_predictions_pct': np.mean(deviations < 0) * 100
#     }

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def comprehensive_model_evaluation(model, model_save_path, test_loader, 
                                 sensor_channels=list(range(1, 22)),
                                 n_engines=6, figsize=(16, 20)):
    """
    Comprehensive evaluation and visualization of the enhanced LSTM autoencoder model.
    
    Args:
        model: The model instance to evaluate
        model_save_path (str): Path to the saved model checkpoint
        test_loader: DataLoader for test data
        sensor_channels (list): List of sensor channels to visualize
        n_engines (int): Number of engines to display in multi-engine plot
        figsize (tuple): Figure size for the plots
    
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load best model - fix the loading issue
    checkpoint = torch.load(model_save_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If the checkpoint is just the state_dict directly (as saved by torch.save(model.state_dict(), path))
        model.load_state_dict(checkpoint)
    
    model.to(device)
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

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.25,
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

    # Calculate and display metrics
    mse = np.mean((pred_valid_detailed - target_valid_detailed) ** 2)
    mae = np.mean(np.abs(pred_valid_detailed - target_valid_detailed))

    ax2.text(0.02, 0.98, f'MSE: {mse:.3f}\nMAE: {mae:.3f}',
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

    # Plot 3: Sensor Reconstruction for Selected Channels
    n_sensors = len(sensor_channels)
    n_cols = 3
    n_rows = (n_sensors + n_cols - 1) // n_cols

    gs_sensors = gs[2, :].subgridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)

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

    print("Files: enhanced_rul_prediction_results.png, enhanced_rul_prediction_results.pdf")

    plt.show()

    # Calculate comprehensive evaluation metrics for ALL engines
    print(f"\n{'='*80}")
    print("COMPREHENSIVE MODEL EVALUATION METRICS - ALL ENGINES")
    print(f"{'='*80}")

    # Collect all valid predictions and targets from ALL engines
    all_valid_predictions = []
    all_valid_targets = []
    engine_metrics = []

    # Calculate metrics for each individual engine
    for engine_idx, engine in enumerate(engine_data):
        mask = engine['mask'].astype(bool)
        if np.any(mask):
            engine_pred = engine['prediction'][mask]
            engine_target = engine['target'][mask]
            
            # Individual engine metrics
            engine_deviations = engine_pred - engine_target
            engine_mse = np.mean(engine_deviations ** 2)
            engine_rmse = np.sqrt(engine_mse)
            engine_mae = np.mean(np.abs(engine_deviations))
            
            # Tolerance-based accuracy
            tolerance_mask = (engine_deviations >= -13) & (engine_deviations <= 10)
            engine_accuracy = np.mean(tolerance_mask) * 100
            
            engine_metrics.append({
                'engine_id': engine_idx + 1,
                'rmse': engine_rmse,
                'mae': engine_mae,
                'mse': engine_mse,
                'accuracy': engine_accuracy,
                'n_samples': len(engine_pred),
                'mean_deviation': np.mean(engine_deviations),
                'std_deviation': np.std(engine_deviations)
            })
            
            # Add to overall collections
            all_valid_predictions.extend(engine_pred)
            all_valid_targets.extend(engine_target)

    all_valid_predictions = np.array(all_valid_predictions)
    all_valid_targets = np.array(all_valid_targets)

    # Calculate overall deviation (d_m = predicted - actual)
    deviations = all_valid_predictions - all_valid_targets

    # Overall metrics
    mse = np.mean(deviations ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(deviations))

    # Accuracy (Tolerance-Based)
    tolerance_mask = (deviations >= -13) & (deviations <= 10)
    accuracy = np.mean(tolerance_mask) * 100

    # Print overall results
    print(f"OVERALL METRICS (All {len(engine_data)} engines):")
    print(f"Root Mean Square Error (RMSE): {rmse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Mean Square Error (MSE): {mse:.6f}")
    print(f"Accuracy (Tolerance [-13, +10]): {accuracy:.2f}%")
    print(f"Mean Deviation: {np.mean(deviations):.6f}")
    print(f"Standard Deviation: {np.std(deviations):.6f}")
    print(f"Total Samples: {len(all_valid_predictions)}")
    print(f"Late Predictions: {np.mean(deviations > 0) * 100:.2f}%")
    print(f"Early Predictions: {np.mean(deviations < 0) * 100:.2f}%")

    # Print individual engine statistics
    print(f"\nINDIVIDUAL ENGINE METRICS:")
    print(f"{'Engine':<8} {'RMSE':<10} {'MAE':<10} {'Accuracy':<12} {'Samples':<10} {'Mean Dev':<12}")
    print(f"{'-'*70}")
    
    for metrics in engine_metrics:
        print(f"{metrics['engine_id']:<8} {metrics['rmse']:<10.4f} {metrics['mae']:<10.4f} "
              f"{metrics['accuracy']:<12.2f} {metrics['n_samples']:<10} {metrics['mean_deviation']:<12.4f}")

    # Calculate summary statistics across engines
    engine_rmses = [m['rmse'] for m in engine_metrics]
    engine_maes = [m['mae'] for m in engine_metrics]
    engine_accuracies = [m['accuracy'] for m in engine_metrics]

    print(f"\nENGINE-WISE STATISTICS:")
    print(f"RMSE - Mean: {np.mean(engine_rmses):.4f}, Std: {np.std(engine_rmses):.4f}, "
          f"Min: {np.min(engine_rmses):.4f}, Max: {np.max(engine_rmses):.4f}")
    print(f"MAE  - Mean: {np.mean(engine_maes):.4f}, Std: {np.std(engine_maes):.4f}, "
          f"Min: {np.min(engine_maes):.4f}, Max: {np.max(engine_maes):.4f}")
    print(f"ACC  - Mean: {np.mean(engine_accuracies):.2f}%, Std: {np.std(engine_accuracies):.2f}%, "
          f"Min: {np.min(engine_accuracies):.2f}%, Max: {np.max(engine_accuracies):.2f}%")

    print(f"{'='*80}")

    return {
        'rmse': rmse,
        'accuracy': accuracy,
        'mae': mae,
        'mse': mse,
        'mean_deviation': np.mean(deviations),
        'std_deviation': np.std(deviations),
        'n_samples': len(all_valid_predictions),
        'n_engines': len(engine_data),
        'late_predictions_pct': np.mean(deviations > 0) * 100,
        'early_predictions_pct': np.mean(deviations < 0) * 100,
        'engine_metrics': engine_metrics,
        'engine_rmse_stats': {
            'mean': np.mean(engine_rmses),
            'std': np.std(engine_rmses),
            'min': np.min(engine_rmses),
            'max': np.max(engine_rmses)
        },
        'engine_mae_stats': {
            'mean': np.mean(engine_maes),
            'std': np.std(engine_maes),
            'min': np.min(engine_maes),
            'max': np.max(engine_maes)
        },
        'engine_accuracy_stats': {
            'mean': np.mean(engine_accuracies),
            'std': np.std(engine_accuracies),
            'min': np.min(engine_accuracies),
            'max': np.max(engine_accuracies)
        }
    }
