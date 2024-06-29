import pandas as pd
import os

# Load the IMDb dataset
file_path = 'IMDB-Dataset-Edited.csv'
df = pd.read_csv(file_path)


# Function to create an imbalanced dataset
def create_imbalanced_dataset(df_, num_pos_train, num_neg_train, num_pos_test, num_neg_test):
    """
    Create an imbalanced dataset by undersampling the majority class and oversampling the minority class.

    Args:
        df_ (pd.DataFrame): Original DataFrame.
        num_pos_train (int): Number of positive samples for training.
        num_neg_train (int): Number of negative samples for training.
        num_pos_test (int): Number of positive samples for testing.
        num_neg_test (int): Number of negative samples for testing.

    Returns:
        pd.DataFrame: Training and testing DataFrames.
    """
    majority_class = 'negative'
    minority_class = 'positive'

    majority_df = df_[df_['sentiment'] == majority_class]
    minority_df = df_[df_['sentiment'] == minority_class]

    # Sample training data
    train_majority_df = majority_df.sample(n=num_neg_train, random_state=42)
    train_minority_df = minority_df.sample(n=num_pos_train, random_state=42)
    train_df_ = pd.concat([train_majority_df, train_minority_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Sample test data
    test_majority_df = majority_df.sample(n=num_neg_test, random_state=42)
    test_minority_df = minority_df.sample(n=num_pos_test, random_state=42)
    test_df_ = pd.concat([test_majority_df, test_minority_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    return train_df_, test_df_


# Create directory for saving imbalanced datasets
output_dir = 'imbalanced_datasets'
os.makedirs(output_dir, exist_ok=True)

# Set the number of samples for each imbalance ratio and save datasets
settings = [
    {"imbalance_ratio": "10%", "num_pos_train": 1250, "num_neg_train": 12000, "num_pos_test": 12500,
     "num_neg_test": 12500},
    {"imbalance_ratio": "5%", "num_pos_train": 625, "num_neg_train": 12000, "num_pos_test": 12500,
     "num_neg_test": 12500},
    {"imbalance_ratio": "4%", "num_pos_train": 500, "num_neg_train": 12000, "num_pos_test": 12500,
     "num_neg_test": 12500},
    {"imbalance_ratio": "2%", "num_pos_train": 250, "num_neg_train": 12000, "num_pos_test": 12500,
     "num_neg_test": 12500},
    {"imbalance_ratio": "1%", "num_pos_train": 125, "num_neg_train": 12000, "num_pos_test": 12500,
     "num_neg_test": 12500}
]

for setting in settings:
    print(f"Creating dataset with Imbalance Ratio: {setting['imbalance_ratio']}")

    train_df, test_df = create_imbalanced_dataset(df,
                                                  setting['num_pos_train'],
                                                  setting['num_neg_train'],
                                                  setting['num_pos_test'],
                                                  setting['num_neg_test'])

    # Print the actual imbalance ratio in the created dataset
    train_class_counts = train_df['sentiment'].value_counts()
    test_class_counts = test_df['sentiment'].value_counts()
    actual_train_imbalance_ratio = train_class_counts.max() / train_class_counts.min()
    print(
        f"Training set - positive reviews: {train_class_counts['positive']} negative reviews: {train_class_counts['negative']}")
    print(
        f"Test set - positive reviews: {test_class_counts['positive']} negative reviews: {test_class_counts['negative']}")
    print(f"Actual Training Imbalance Ratio: {actual_train_imbalance_ratio}")

    # Save the imbalanced dataset to CSV files
    imbalanced_train_file = os.path.join(output_dir, f'IMDB_Train_Dataset_Imbalance_{setting["imbalance_ratio"]}.csv')
    imbalanced_test_file = os.path.join(output_dir, f'IMDB_Test_Dataset_Balanced.csv')

    train_df.to_csv(imbalanced_train_file, index=False)
    test_df.to_csv(imbalanced_test_file, index=False)

    print(f"Saved imbalanced training dataset to {imbalanced_train_file}")
    print(f"Saved balanced test dataset to {imbalanced_test_file}")

print("All imbalanced datasets created and saved successfully.")
