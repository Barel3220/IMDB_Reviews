import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from single_dqn_agent import SingleDQNAgent
from single_dqn_reply_buffer import device

# Initialize tokenizer
tokenizer = get_tokenizer('basic_english')


def predict_sentiment(review, single_dqn_agent, vocab, max_len=500):
    """
    Predict the sentiment of a given review.

    Args:
        review (str): The review text.
        single_dqn_agent (SingleDQNAgent): The trained agent.
        vocab (dict): The vocabulary used by the agent.
        max_len (int): Maximum length of the tokenized sequence.

    Returns:
        str: The predicted sentiment ('positive' or 'negative').
    """

    single_dqn_agent.qnetwork.eval()
    tokens = tokenizer(review)
    token_ids = [vocab.get(token, vocab['<pad>']) for token in tokens]
    token_ids = torch.tensor(token_ids, dtype=torch.long).to(device)

    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    else:
        token_ids = torch.cat([token_ids, torch.tensor([vocab['<pad>']] * (max_len - len(token_ids))).to(device)])

    with torch.no_grad():
        output = single_dqn_agent.qnetwork(token_ids.unsqueeze(0))
        prediction = torch.argmax(output, dim=1).item()

    sentiment = 'positive' if prediction == 1 else 'negative'
    return sentiment


def predict_sentiments(small_loader, single_dqn_agent, train_dataset, max_len=500):
    """
    Predict the sentiment of reviews from a file and print the predicted vs. true sentiments.

    Args:
        file_path (str): Path to the CSV file containing reviews.
        single_dqn_agent (SingleDQNAgent): The trained agent.
        vocab (dict): The vocabulary used by the agent.
        max_len (int): Maximum length of the tokenized sequence.
    """
    # df = pd.read_csv(file_path)

    print(f"{'Review':<80} {'True':<10} {'Predicted':<10}")
    print("=" * 100)

    for text, label in small_loader:
        review = " ".join(
            [train_dataset.vocab.itos[token.item()] for token in text[0] if token != train_dataset.vocab["<pad>"]])
        true_sentiment = label.item()
        predicted_sentiment = predict_sentiment(review, single_dqn_agent, train_dataset.vocab, max_len=500)
        print("finished")
        print(f"{review[:77]:<80} {true_sentiment:<10} {predicted_sentiment:<10}")


    # correct = 0
    #
    # for review, true_sentiment in zip(df['review'], df['sentiment']):
    #     predicted_sentiment = predict_sentiment(review, single_dqn_agent, vocab, max_len)
    #     true_sentiment_label = 'positive' if true_sentiment == 1 else 'negative'
    #     correct += 1 if true_sentiment_label == predicted_sentiment else 0
    #     print(f"{review[:77]:<80} {true_sentiment_label:<10} {predicted_sentiment:<10}")
    #
    # print(f"The results are {correct} / 10 correct")
