import numpy as np
import pandas as pd
import plotly.graph_objects as go

def plot_rul_degradation_3d(train_data):
    """
    Create a 3D line plot showing RUL degradation curves for all C-MAPSS engine subsets.
    
    Parameters:
    train_data (dict): Dictionary containing training data for each subset (FD001, FD002, FD003, FD004)
    
    Returns:
    plotly.graph_objects.Figure: 3D line plot figure
    """
    # === Prepare data from all subsets ===
    fig = go.Figure()
    
    # Color map for different subsets
    colors = {'FD001': 'blue', 'FD002': 'red', 'FD003': 'green', 'FD004': 'orange'}

    for subset_key in ['FD001', 'FD002', 'FD003', 'FD004']:
        data = train_data[subset_key]
        unit_ids = np.unique(data[:, 0])

        for unit_id in unit_ids:
            unit_data = data[data[:, 0] == unit_id]
            cycles = unit_data[:, 1]
            rul = np.flip(np.arange(len(cycles)))  # decreasing RUL
            
            # Add each engine as a separate trace to avoid connecting between engines
            fig.add_trace(go.Scatter3d(
                x=cycles,
                y=[int(unit_id)] * len(cycles),
                z=rul,
                mode='lines',
                line=dict(color=colors[subset_key], width=3),
                name=subset_key,
                showlegend=bool(unit_id == unit_ids[0]),  # Convert numpy bool to Python bool
                legendgroup=subset_key
            ))

    # === Beautify ===
    fig.update_layout(
        title='C-MAPSS Engine RUL Degradation Curves',
        width=1000,
        height=800,
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True,
        legend_title_text='Subset',
        template='simple_white',
        scene=dict(
            camera=dict(
                eye=dict(x=1.8, y=-1.2, z=0.8)
            ),
            xaxis=dict(title='Operation Cycles', backgroundcolor='white', showbackground=False, showgrid=False, zeroline=False),
            yaxis=dict(title='Engine ID', backgroundcolor='white', showbackground=False, showgrid=False, zeroline=False),
            zaxis=dict(title='Remaining useful life', backgroundcolor='white', showbackground=False, showgrid=False, zeroline=False),
        )
    )

    # === Print interpretation information ===
    print("🔍 Interpretation of the Plot:")
    print()
    print("This 3D plot shows RUL degradation curves for engines in the C-MAPSS dataset.")
    print()
    print("Axes represent:")
    print("X: Operation Cycles")
    print("Y: Engine ID")
    print("Z: Remaining Useful Life (RUL)")
    print()
    print("Different colors indicate four subsets (FD001–FD004), each with distinct operating conditions and fault modes.")
    print()
    print("⚠️ Key Prognostics Challenges:")
    print()
    print("Heterogeneous conditions: Each subset exhibits unique degradation patterns, making generalization difficult.")
    print()
    print("Varying initial RUL: Engines start with different lifespans, complicating model alignment.")
    print()
    print("Nonlinear and irregular degradation: RUL decreases at different rates, some with sharp drops or flat regions.")
    print()
    print("Multi-domain complexity: Requires models that can adapt across multiple environments and failure types.")

    return fig
