import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_rul_degradation_3d(train_data, n_longest=5, n_shortest=5):
    """
    Create 4 subplots of 3D scatter plots showing RUL degradation curves for selected engines from each C-MAPSS engine subset.
    
    Parameters:
    train_data (dict): Dictionary containing training data for each subset (FD001, FD002, FD003, FD004)
    n_longest (int): Number of longest curves to select from each subset
    n_shortest (int): Number of shortest curves to select from each subset
    
    Returns:
    plotly.graph_objects.Figure: 3D scatter plot figure with 4 subplots
    """
    # === Validate parameters ===
    subset_info = {}
    for subset_key in ['FD001', 'FD002', 'FD003', 'FD004']:
        data = train_data[subset_key]
        unit_ids = np.unique(data[:, 0])
        total_engines = len(unit_ids)
        subset_info[subset_key] = total_engines
        
        if n_longest > total_engines or n_shortest > total_engines:
            error_msg = f"Parameter error: n_longest ({n_longest}) or n_shortest ({n_shortest}) exceeds available engines.\n"
            error_msg += "Available ranges for each subset:\n"
            for key, count in subset_info.items():
                error_msg += f"{key}: 1 to {count} engines\n"
            print(error_msg)
            return None
    
    # === Create subplots ===
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}],
               [{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=['FD001', 'FD002', 'FD003', 'FD004']
    )
    
    # Color map for different subsets
    colors = {'FD001': 'blue', 'FD002': 'red', 'FD003': 'green', 'FD004': 'orange'}
    
    # Subplot positions
    subplot_positions = {
        'FD001': (1, 1),
        'FD002': (1, 2),
        'FD003': (2, 1),
        'FD004': (2, 2)
    }

    for subset_key in ['FD001', 'FD002', 'FD003', 'FD004']:
        data = train_data[subset_key]
        unit_ids = np.unique(data[:, 0])
        row, col = subplot_positions[subset_key]
        
        # Calculate cycle lengths for each engine
        engine_lengths = []
        for unit_id in unit_ids:
            unit_data = data[data[:, 0] == unit_id]
            cycles = len(unit_data)
            engine_lengths.append((unit_id, cycles))
        
        # Sort by cycle length
        engine_lengths.sort(key=lambda x: x[1])
        
        # Select shortest and longest engines
        selected_engines = []
        if n_shortest > 0:
            selected_engines.extend([x[0] for x in engine_lengths[:n_shortest]])
        if n_longest > 0:
            selected_engines.extend([x[0] for x in engine_lengths[-n_longest:]])
        
        # Remove duplicates while preserving order
        selected_engines = list(dict.fromkeys(selected_engines))

        for unit_id in selected_engines:
            unit_data = data[data[:, 0] == unit_id]
            cycles = unit_data[:, 1]
            rul = np.flip(np.arange(len(cycles)))  # decreasing RUL
            
            # Calculate operating setting (average of operational settings 1, 2, 3)
            # Assuming columns 2, 3, 4 are operational settings
            operating_setting = np.mean(unit_data[:, 2:5], axis=1)
            
            # Add each engine as a separate trace to avoid connecting between engines
            fig.add_trace(go.Scatter3d(
                x=cycles,
                y=rul,
                z=operating_setting,
                mode='markers',
                marker=dict(color=colors[subset_key], size=4),
                name=f'{subset_key} Engine {int(unit_id)}',
                showlegend=False,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Operation Cycles: %{x}<br>' +
                              'Remaining Useful Life: %{y}<br>' +
                              'Operating Setting: %{z:.2f}<br>' +
                              '<extra></extra>'
            ), row=row, col=col)

    # === Beautify ===
    fig.update_layout(
        title=f'C-MAPSS Engine RUL Degradation Curves by Subset (Top {n_longest} Longest + Top {n_shortest} Shortest)',
        width=1200,
        height=1000,
        margin=dict(l=0, r=0, t=80, b=0),
        showlegend=False,
        template='simple_white'
    )
    
    # Update scene properties for each subplot
    for i, subset_key in enumerate(['FD001', 'FD002', 'FD003', 'FD004']):
        scene_name = f'scene{i+1}' if i > 0 else 'scene'
        fig.update_layout(**{
            scene_name: dict(
                camera=dict(
                    eye=dict(x=1.5, y=-1.5, z=1.2)
                ),
                xaxis=dict(title='Operation Cycles', backgroundcolor='white', showbackground=False, showgrid=True, zeroline=False),
                yaxis=dict(title='Remaining Useful Life', backgroundcolor='white', showbackground=False, showgrid=True, zeroline=False),
                zaxis=dict(title='Operating Setting', backgroundcolor='white', showbackground=False, showgrid=True, zeroline=False),
            )
        })

    # === Print interpretation information ===
    print("🔍 Interpretation of the Plot:")
    print()
    print("This plot shows 4 subplots, each representing one C-MAPSS subset with RUL degradation curves.")
    print(f"Selected: {n_longest} longest and {n_shortest} shortest curves from each subset.")
    print()
    print("Axes represent:")
    print("X: Operation Cycles")
    print("Y: Remaining Useful Life (RUL)")
    print("Z: Operating Setting (average of operational settings)")
    print()
    print("Each subplot shows a different subset (FD001–FD004) with distinct operating conditions and fault modes.")
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



# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go

# def plot_rul_degradation_3d(train_data, n_longest=5, n_shortest=5):
#     """
#     Create a 3D line plot showing RUL degradation curves for selected engines from all C-MAPSS engine subsets.
    
#     Parameters:
#     train_data (dict): Dictionary containing training data for each subset (FD001, FD002, FD003, FD004)
#     n_longest (int): Number of longest curves to select from each subset
#     n_shortest (int): Number of shortest curves to select from each subset
    
#     Returns:
#     plotly.graph_objects.Figure: 3D line plot figure
#     """
#     # === Validate parameters ===
#     subset_info = {}
#     for subset_key in ['FD001', 'FD002', 'FD003', 'FD004']:
#         data = train_data[subset_key]
#         unit_ids = np.unique(data[:, 0])
#         total_engines = len(unit_ids)
#         subset_info[subset_key] = total_engines
        
#         if n_longest > total_engines or n_shortest > total_engines:
#             error_msg = f"Parameter error: n_longest ({n_longest}) or n_shortest ({n_shortest}) exceeds available engines.\n"
#             error_msg += "Available ranges for each subset:\n"
#             for key, count in subset_info.items():
#                 error_msg += f"{key}: 1 to {count} engines\n"
#             print(error_msg)
#             return None
    
#     # === Prepare data from all subsets ===
#     fig = go.Figure()
    
#     # Color map for different subsets
#     colors = {'FD001': 'blue', 'FD002': 'red', 'FD003': 'green', 'FD004': 'orange'}

#     for subset_key in ['FD001', 'FD002', 'FD003', 'FD004']:
#         data = train_data[subset_key]
#         unit_ids = np.unique(data[:, 0])
        
#         # Calculate cycle lengths for each engine
#         engine_lengths = []
#         for unit_id in unit_ids:
#             unit_data = data[data[:, 0] == unit_id]
#             cycles = len(unit_data)
#             engine_lengths.append((unit_id, cycles))
        
#         # Sort by cycle length
#         engine_lengths.sort(key=lambda x: x[1])
        
#         # Select shortest and longest engines
#         selected_engines = []
#         if n_shortest > 0:
#             selected_engines.extend([x[0] for x in engine_lengths[:n_shortest]])
#         if n_longest > 0:
#             selected_engines.extend([x[0] for x in engine_lengths[-n_longest:]])
        
#         # Remove duplicates while preserving order
#         selected_engines = list(dict.fromkeys(selected_engines))

#         for unit_id in selected_engines:
#             unit_data = data[data[:, 0] == unit_id]
#             cycles = unit_data[:, 1]
#             rul = np.flip(np.arange(len(cycles)))  # decreasing RUL
            
#             # Add each engine as a separate trace to avoid connecting between engines
#             fig.add_trace(go.Scatter3d(
#                 x=cycles,
#                 y=[int(unit_id)] * len(cycles),
#                 z=rul,
#                 mode='lines',
#                 line=dict(color=colors[subset_key], width=3),
#                 name=subset_key,
#                 showlegend=bool(unit_id == selected_engines[0]),  # Convert numpy bool to Python bool
#                 legendgroup=subset_key,
#                 hovertemplate='<b>%{fullData.name}</b><br>' +
#                               'Operation Cycles: %{x}<br>' +
#                               'Engine ID: %{y}<br>' +
#                               'Remaining Useful Life: %{z}<br>' +
#                               '<extra></extra>'
#             ))

#     # === Beautify ===
#     fig.update_layout(
#         title=f'C-MAPSS Engine RUL Degradation Curves (Top {n_longest} Longest + Top {n_shortest} Shortest per Subset)',
#         width=1000,
#         height=800,
#         margin=dict(l=0, r=0, t=50, b=0),
#         showlegend=True,
#         legend_title_text='Subset',
#         template='simple_white',
#         scene=dict(
#             camera=dict(
#                 eye=dict(x=1.8, y=-1.2, z=0.8)
#             ),
#             xaxis=dict(title='Operation Cycles', backgroundcolor='white', showbackground=False, showgrid=False, zeroline=False),
#             yaxis=dict(title='Engine ID', backgroundcolor='white', showbackground=False, showgrid=False, zeroline=False),
#             zaxis=dict(title='Remaining useful life', backgroundcolor='white', showbackground=False, showgrid=False, zeroline=False),
#         )
#     )

#     # === Print interpretation information ===
#     print("🔍 Interpretation of the Plot:")
#     print()
#     print("This 3D plot shows RUL degradation curves for selected engines in the C-MAPSS dataset.")
#     print(f"Selected: {n_longest} longest and {n_shortest} shortest curves from each subset.")
#     print()
#     print("Axes represent:")
#     print("X: Operation Cycles")
#     print("Y: Engine ID")
#     print("Z: Remaining Useful Life (RUL)")
#     print()
#     print("Different colors indicate four subsets (FD001–FD004), each with distinct operating conditions and fault modes.")
#     print()
#     print("⚠️ Key Prognostics Challenges:")
#     print()
#     print("Heterogeneous conditions: Each subset exhibits unique degradation patterns, making generalization difficult.")
#     print()
#     print("Varying initial RUL: Engines start with different lifespans, complicating model alignment.")
#     print()
#     print("Nonlinear and irregular degradation: RUL decreases at different rates, some with sharp drops or flat regions.")
#     print()
#     print("Multi-domain complexity: Requires models that can adapt across multiple environments and failure types.")

#     return fig
