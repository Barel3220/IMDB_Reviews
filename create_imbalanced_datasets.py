import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# Load the IMDb dataset
file_path = 'IMDB-Dataset-Edited.csv'
df = pd.read_csv(file_path)

# Encode the sentiments (positive/negative) as integers
le = LabelEncoder()
df['sentiment'] = le.fit_transform(df['sentiment'])


# Function to create an imbalanced dataset
def create_imbalanced_dataset(df_, imbalance_ratio_):
    class_counts_ = df_['sentiment'].value_counts()
    majority_class = class_counts_.idxmax()
    minority_class = class_counts_.idxmin()

    print(f"Original class counts: {class_counts_}")
    print(f"Majority class: {majority_class}, Minority class: {minority_class}")

    majority_df = df_[df_['sentiment'] == majority_class]
    minority_df = df_[df_['sentiment'] == minority_class]

    majority_size = len(majority_df)
    # Calculate the new majority size by adding the given percentage of the minority class size
    new_majority_size = int(majority_size * (1 + imbalance_ratio_))

    # Ensure we don't exceed the available number of minority class samples
    if new_majority_size > len(minority_df) + majority_size:
        new_majority_size = len(minority_df) + majority_size

    print(f"New majority size: {new_majority_size}, Minority size: {majority_size}")

    majority_df = majority_df.sample(n=majority_size, random_state=42)
    additional_majority_samples = majority_df.sample(n=new_majority_size - majority_size, replace=True, random_state=42)

    imbalanced_df_ = pd.concat([majority_df, additional_majority_samples, minority_df]).sample(frac=1,
                                                                                               random_state=42).reset_index(
        drop=True)

    # Print the class counts of the created imbalanced dataset
    imbalanced_class_counts = imbalanced_df_['sentiment'].value_counts()
    print(f"Imbalanced class counts: {imbalanced_class_counts}")

    return imbalanced_df_


# Create directory for saving imbalanced datasets
output_dir = 'imbalanced_datasets'
os.makedirs(output_dir, exist_ok=True)

# Set imbalance ratios and save datasets
imbalance_ratios = [0.1, 0.05, 0.04, 0.02, 0.01]  # These represent the desired imbalance ratios
for imbalance_ratio in imbalance_ratios:
    print(f"Creating dataset with Imbalance Ratio: {imbalance_ratio}")

    imbalanced_df = create_imbalanced_dataset(df, imbalance_ratio)

    # Print the actual imbalance ratio in the created dataset
    class_counts = imbalanced_df['sentiment'].value_counts()
    actual_imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"Actual Imbalance Ratio: {actual_imbalance_ratio:.2f}")

    # Save the imbalanced dataset to a CSV file
    output_file = os.path.join(output_dir, f'IMDB_Dataset_Imbalance_{imbalance_ratio:.2f}.csv')
    imbalanced_df.to_csv(output_file, index=False)
    print(f"Saved imbalanced dataset to {output_file}")

print("All imbalanced datasets created and saved successfully.")
