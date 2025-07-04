import torch
import numpy as np
import plotly.graph_objects as go

def show_results(model, test_loader, device='cpu', model_path='best_lightweight_model.pth', num_engines=5, reset_threshold=600):
    """
    Function to display model test results, including evaluation metrics and visualization charts
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Computing device
        model_path: Model file path
        num_engines: Number of engines to visualize
        reset_threshold: Threshold for detecting engine restart
    """
    # === Model Evaluation ===
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y, *_ in test_loader:
            X = X.to(device)
            preds = model(X).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    d = y_pred - y_true

    # RMSE
    rmse = np.sqrt(np.mean(d ** 2))

    # Score (asymmetric penalty)
    score = np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))

    # Accuracy (within tolerance range)
    acc = np.mean((d >= -13) & (d <= 10)) * 100

    # Print evaluation results
    print(f"✅ Test RMSE: {rmse:.2f}")
    print(f"✅ Test Score: {score:.2f}")
    print(f"✅ Accuracy (Tolerance [-13, +10]): {acc:.2f}%")

    # === Plot RUL Comparison Chart ===
    # Detect engine start indices
    starts = [0]
    for i in range(1, len(y_true)):
        if y_true[i] > y_true[i - 1] and y_true[i] > reset_threshold:
            starts.append(i)
    starts.append(len(y_true))  # Add final boundary

    fig = go.Figure()

    # Plot data for each engine
    for engine_id in range(min(num_engines, len(starts) - 1)):
        start = starts[engine_id]
        end = starts[engine_id + 1]
        true_seg = np.array(y_true[start:end])
        pred_seg = np.array(y_pred[start:end])

        # Split by monotonicity breakpoints
        diffs = np.diff(true_seg)
        break_points = np.where(diffs > 0)[0] + 1
        segments = np.split(np.arange(len(true_seg)), break_points)

        # Add line traces for each segment
        for seg in segments:
            if len(seg) < 2:
                continue
            x_vals = (seg + start).tolist()  # Adjust x-axis to global index
            
            # True RUL - Use blue dashed line consistently
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=true_seg[seg],
                mode='lines',
                name="True RUL" if engine_id == 0 and seg[0] == 0 else None,
                line=dict(dash='dash', color='blue'),
                opacity=0.8,
                showlegend=bool(engine_id == 0 and seg[0] == 0),
                legendgroup="true"
            ))
            
            # Predicted RUL - Use red solid line consistently
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=pred_seg[seg],
                mode='lines',
                name="Predicted RUL" if engine_id == 0 and seg[0] == 0 else None,
                line=dict(color='red'),
                opacity=0.8,
                showlegend=bool(engine_id == 0 and seg[0] == 0),
                legendgroup="pred"
            ))

    # Update chart layout
    fig.update_layout(
        title=f"Predicted vs True RUL (First {num_engines} Engines, Segmented)",
        xaxis_title="Cycle Index",
        yaxis_title="RUL",
        legend_title="Legend",
        template="plotly_white",
        height=500,
        width=1000
    )
    fig.show()
    
    return rmse, score, acc, y_true, y_pred
