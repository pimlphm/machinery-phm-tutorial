from load_turbofan_data import load_turbofan_data

datasets = load_turbofan_data("/content/turbofan_data")

for fd, data in datasets.items():
    print(f"{fd}: Train={data['train_data'].shape}, Test={data['test_data'].shape}")
