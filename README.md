# IMDB Sentiment Analysis with Double DQN

This project implements a sentiment analysis system for IMDB movie reviews using a Double Deep Q-Network (Double DQN). 
The Double DQN is implemented through the Agent class, where two Q-Networks (qnetwork_local and qnetwork_target) are initialized. 
The system processes reviews to classify them as positive or negative based on their content.

## Project Structure

- **imdb_dataset.py:** Contains the `IMDBDataset` class for processing the IMDB dataset.
- **qnetwork.py:** Defines the `QNetwork` class, a neural network model used by the agent.
- **replay_buffer.py:** Implements the `ReplayBuffer` class for experience replay.
- **agent.py:** Contains the `Agent` class that interacts with the environment, updates the Q-Network, and manages the training process.
- **train.py:** Defines the training loop for the agent.
- **evaluate.py:** Contains the evaluation function to assess the agent's performance.
- **predict.py:** Implements functions for predicting sentiments of reviews and printing the predicted vs. true sentiments.
- **main.py:** The main script to train, evaluate, and predict using the trained model. It also saves the last 10 reviews for prediction.

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
```

## Usage

### Training the Model

To train the model, run the `main.py` script. This will train the agent using the IMDB dataset and save the last 10 reviews for prediction:

```sh python main.py```

### Evaluating the Model

The evaluation function is included in the main.py script and will be called automatically after training. It evaluates the trained agent on the test dataset and prints the accuracy.

### Predicting Sentiments

To predict the sentiment of the last 10 reviews saved during the training process, the predict.py script will be executed by the main.py script. It reads the reviews from last_10_reviews.csv and prints the predicted vs. true sentiments.

### File Descriptions

	•	IMDB-Dataset.csv: The dataset file containing the IMDB movie reviews and their corresponding sentiments.
	•	last_10_reviews.csv: The file where the last 10 reviews are saved for prediction purposes.
	•	requirements.txt: Contains the list of required Python packages for the project.