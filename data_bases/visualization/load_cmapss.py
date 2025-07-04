
# === Load and Prepare CMAPSS Data ===
def load_cmapss(base_path, dataset=None):
    # Define column names for the CMAPSS data files (3 operational settings + 21 sensors)
    cols = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    # If a specific dataset is requested, process only that one
    if dataset is not None:
        # Extract dataset number from name (e.g., 'FD001' -> 1)
        dataset_num = int(dataset[-1])
        
        # Load training data for specified dataset
        train = pd.read_csv(f"{base_path}/train_{dataset}.txt", sep='\s+', header=None, names=cols)

        # Load test data and the corresponding RUL (remaining useful life) ground truth values
        test = pd.read_csv(f"{base_path}/test_{dataset}.txt", sep='\s+', header=None, names=cols)
        rul = pd.read_csv(f"{base_path}/RUL_{dataset}.txt", sep='\s+', header=None, names=['true_RUL'])

        # Calculate the RUL for training data by subtracting each cycle from its engine's max cycle
        max_cycle = train.groupby('engine_id')['cycle'].transform('max')  # Each engine's max cycle
        train['RUL'] = max_cycle - train['cycle']  # RUL = max_cycle - current_cycle
        train['dataset'] = dataset_num  # Label the subset ID

        # Compute max cycle per engine in test set to determine current age
        test['max_cycle'] = test.groupby('engine_id')['cycle'].transform('max')

        # Assign RUL to each time step of each test engine based on true_RUL at final cycle
        test['RUL'] = test.apply(
            lambda row: rul.iloc[int(row['engine_id']) - 1, 0] + (row['max_cycle'] - row['cycle']),
            axis=1
        )

        # Drop the temporary max_cycle column
        test.drop('max_cycle', axis=1, inplace=True)
        test['dataset'] = dataset_num  # Add subset label

        # Clip RUL values: remove bottom 5% and top 5% of RUL values
        train_rul_quantiles = train['RUL'].quantile([0.05, 0.95])
        train = train[(train['RUL'] >= train_rul_quantiles[0.05]) & (train['RUL'] <= train_rul_quantiles[0.95])]
        
        test_rul_quantiles = test['RUL'].quantile([0.05, 0.95])
        test = test[(test['RUL'] >= test_rul_quantiles[0.05]) & (test['RUL'] <= test_rul_quantiles[0.95])]

        return train, test
    
    # Original behavior: load all datasets if no specific dataset is requested
    # Initialize empty DataFrames to hold combined training and test data from all four subdatasets (FD001–FD004)
    train_all, test_all = pd.DataFrame(), pd.DataFrame()

    # Iterate through all four CMAPSS subsets (FD001 to FD004)
    for i in range(1, 5):
        # Load training data for current subset
        train = pd.read_csv(f"{base_path}/train_FD00{i}.txt", sep='\s+', header=None, names=cols)

        # Load test data and the corresponding RUL (remaining useful life) ground truth values
        test = pd.read_csv(f"{base_path}/test_FD00{i}.txt", sep='\s+', header=None, names=cols)
        rul = pd.read_csv(f"{base_path}/RUL_FD00{i}.txt", sep='\s+', header=None, names=['true_RUL'])

        # Calculate the RUL for training data by subtracting each cycle from its engine's max cycle
        max_cycle = train.groupby('engine_id')['cycle'].transform('max')  # Each engine's max cycle
        train['RUL'] = max_cycle - train['cycle']  # RUL = max_cycle - current_cycle
        train['dataset'] = i  # Label the subset ID (1 to 4)
        train_all = pd.concat([train_all, train], ignore_index=True)  # Append to the global training set

        # Compute max cycle per engine in test set to determine current age
        test['max_cycle'] = test.groupby('engine_id')['cycle'].transform('max')

        # To prevent duplicate engine IDs across different subsets, shift test engine IDs by offset
        offset = test_all['engine_id'].max() + 1 if not test_all.empty else 0
        test['engine_id'] += offset  # Shift engine_id to ensure uniqueness

        # Assign RUL to each time step of each test engine based on true_RUL at final cycle
        test['RUL'] = test.apply(
            lambda row: rul.iloc[int(row['engine_id']) - 1 - offset, 0] + (row['max_cycle'] - row['cycle']),
            axis=1
        )

        # Drop the temporary max_cycle column
        test.drop('max_cycle', axis=1, inplace=True)
        test['dataset'] = i  # Add subset label
        test_all = pd.concat([test_all, test], ignore_index=True)  # Append to the global test set

    # Clip RUL values for combined datasets: remove bottom 5% and top 5% of RUL values
    train_rul_quantiles = train_all['RUL'].quantile([0.05, 0.95])
    train_all = train_all[(train_all['RUL'] >= train_rul_quantiles[0.05]) & (train_all['RUL'] <= train_rul_quantiles[0.95])]
    
    test_rul_quantiles = test_all['RUL'].quantile([0.05, 0.95])
    test_all = test_all[(test_all['RUL'] >= test_rul_quantiles[0.05]) & (test_all['RUL'] <= test_rul_quantiles[0.95])]

    # Return concatenated training and test DataFrames covering FD001 to FD004
    return train_all, test_all
