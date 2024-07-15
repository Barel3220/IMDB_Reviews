# IMDB Sentiment Analysis with DQN and Double DQN

This project implements a sentiment analysis system for IMDB movie reviews using Single Deep Q-Network (Single DQN) and Double Deep Q-Network (Double DQN). The systems process reviews to classify them as positive or negative based on their content.

<b>Written and executed by Angelina (Lina) Rozentsvig and Barel Hatuka</b>

## Project Structure

### Single DQN

- **imdb_dataset.py:** Contains the `IMDBDataset` class for processing the IMDB dataset.
- **single_dqn/single_dqn_agent.py:** Contains the `SingleDQNAgent` class that interacts with the environment, updates the Q-Network, and manages the training process.
- **single_dqn/single_dqn_train.py:** Defines the training loop for the Single DQN agent.
- **single_dqn/single_dqn_evaluate.py:** Contains the evaluation function to assess the Single DQN agent's performance.
- **single_dqn/single_dqn_main.py:** The main script to train, evaluate, and predict using the trained Single DQN model.

### Double DQN

- **imdb_dataset.py:** Contains the `IMDBDataset` class for processing the IMDB dataset.
- **double_dqn/agent.py:** Contains the `DoubleDQNAgent` class that interacts with the environment, updates the Q-Network, and manages the training process.
- **double_dqn/train.py:** Defines the training loop for the Double DQN agent.
- **double_dqn/evaluate.py:** Contains the evaluation function to assess the Double DQN agent's performance.
- **double_dqn/main.py:** The main script to train, evaluate, and predict using the trained Double DQN model.

### Shared Files

- **replay_buffer.py:** Implements the `ReplayBuffer` class for experience replay used by both Single DQN and Double DQN.
- **text_model.py:** Defines the `TextModel` class, a neural network model used by both Single DQN and Double DQN.
- **plotter.py:** Contains functions to plot the training metrics.
- **predict.py:** Implements functions for predicting sentiments of reviews and printing the predicted vs. true sentiments.
- **plot_data_statistics.py:** Contains functions to plot data statistics such as label distribution, review length distribution, and class imbalance ratio.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone [repository_url]
    ```
2. Navigate to the project directory:
    ```sh
    cd IMDB_Reviews
    ```
3. Create a virtual environment:
    ```sh
    python -m venv env
    ```
4. Activate the virtual environment:
    - **Windows:**
        ```sh
        .\env\Scripts\activate
        ```
    - **Mac/Linux:**
        ```sh
        source env/bin/activate
        ```
5. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

requirements.txt to install:
```
pandas==1.3.5
scikit-learn==1.0.2
torch==1.11.0
torchtext==0.12.0
tqdm==4.62.3
wordcloud==1.8.1
matplotlib==3.4.3
seaborn==0.11.2
```

## Usage

### Training the Single DQN Model

To train the Single DQN model, run the `single_dqn/single_dqn_main.py` script. This will train the agent using the IMDB dataset and save the last 10 reviews for prediction:

```sh
python single_dqn/single_dqn_main.py
```

### Training the Double DQN Model

To train the Double DQN model, run the double_dqn/main.py script. This will train the agent using the IMDB dataset and save the last 10 reviews for prediction:

```sh
python double_dqn/main.py
```

### Evaluating the Model

The evaluation functions are included in the respective main.py scripts and will be called automatically after training. They evaluate the trained agents on the test dataset and print the accuracy.

### Predicting Sentiments

To predict the sentiment of the last 10 reviews saved during the training process, the respective main.py scripts will execute the predict.py script. It reads the reviews from last_10_reviews.csv and prints the predicted vs. true sentiments.

### Plotting Data Statistics

To plot data statistics, run the plot_data_statistics.py script. This script generates various plots such as label distribution, review length distribution, and class imbalance ratio:

```sh
python plot_data_statistics.py
```

## File Descriptions

- **IMDB-Dataset.csv: The dataset file containing the IMDB movie reviews and their corresponding sentiments.
- **last_10_reviews.csv: The file where the last 10 reviews are saved for prediction purposes.
- **requirements.txt: Contains the list of required Python packages for the project.