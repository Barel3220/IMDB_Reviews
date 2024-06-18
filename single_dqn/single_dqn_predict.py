import torch
from torchtext.data.utils import get_tokenizer
from single_dqn_agent import SingleDQNAgent
from single_dqn_replay_buffer import device

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
