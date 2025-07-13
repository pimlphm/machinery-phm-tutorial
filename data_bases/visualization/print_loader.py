def print_loader_sizes(train_loader, val_loader, test_loader):
    print("\n📦 DataLoader Summary")
    print("=" * 60)

    for name, loader in zip(['Train', 'Validation', 'Test'], [train_loader, val_loader, test_loader]):
        # Total number of samples
        num_batches = len(loader)
        batch = next(iter(loader))
        batch_size = batch['x'].shape[0]
        seq_len = batch['x'].shape[1]
        num_features = batch['x'].shape[2]

        total_samples = num_batches * batch_size  # Approximate

        print(f"🔹 {name} Loader:")
        print(f"    - Batches:        {num_batches}")
        print(f"    - Batch size:     {batch_size}")
        print(f"    - Input shape:    [B={batch_size}, T={seq_len}, C={num_features}]")
        print(f"    - RUL shape:      {batch['rul'].shape}")
        print(f"    - Mask shape:     {batch['mask'].shape}")
        print("-" * 60)
