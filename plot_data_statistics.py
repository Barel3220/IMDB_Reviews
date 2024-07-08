import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# Function to plot pie chart for label distribution
def plot_label_distribution(df, label_column):
    class_counts = df[label_column].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', colors=['skyblue', 'salmon'], startangle=140)
    plt.title('Distribution of Sentiment Labels')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


# Function to plot review length distribution
def plot_review_length_distribution(df, text_column):
    df['review_length'] = df[text_column].apply(lambda x: len(x.split()))
    plt.figure(figsize=(10, 6))
    sns.histplot(df['review_length'], bins=50, kde=True, color='purple')
    plt.title('Distribution of Review Lengths')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.show()


# Function to plot class imbalance ratio
def plot_class_imbalance_ratio(df, label_column):
    class_counts = df[label_column].value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()
    plt.figure(figsize=(8, 6))
    plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', colors=['skyblue', 'salmon'], startangle=140)
    plt.title(f'Class Imbalance Ratio: {imbalance_ratio:.2f}')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


# Function to plot review length by sentiment
def plot_review_length_by_sentiment(df, text_column, label_column):
    df['review_length'] = df[text_column].apply(lambda x: len(x.split()))
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=label_column, y='review_length', data=df, palette=['skyblue', 'salmon'])
    plt.title('Review Length Distribution by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Words')
    plt.show()


def main():
    # Set file paths
    train_file_path = 'imbalanced_datasets/IMDB_Train_Dataset_Imbalance_10%.csv'
    # test_file_path = 'imbalanced_datasets/IMDB_Test_Dataset_Balanced.csv'

    # Load the datasets
    df = pd.read_csv(train_file_path)
    # test_df = pd.read_csv(test_file_path)

    # Concatenate train and test datasets for overall analysis
    # df = pd.concat([train_df, test_df])

    # Plot label distribution
    plot_label_distribution(df, 'sentiment')

    # Plot review length distribution
    plot_review_length_distribution(df, 'review')

    # Plot class imbalance ratio
    plot_class_imbalance_ratio(df, 'sentiment')

    # Plot review length by sentiment
    plot_review_length_by_sentiment(df, 'review', 'sentiment')


if __name__ == "__main__":
    main()
