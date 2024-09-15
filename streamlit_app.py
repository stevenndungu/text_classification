#%%
import streamlit as st
from utils import *
import torch
import pickle5 as pickle

import pandas as pd

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
max_len=401
# Initialize the model and load the saved state
config = TransformerConfig(
          vocab_size=len(vocab),
          hidden_size=256,
          num_attention_heads=8,
          num_hidden_layers=8,
          intermediate_size=512,
          hidden_dropout_prob=0.1,
          max_position_embeddings=401,
          num_labels=2,
          activation_function='gelu'
      )

# Tokenize and convert the review to input IDs
def preprocess_review(review_text, vocab, max_len):
    # Tokenize the review text
    tokens = review_text.lower()
    tokens = re.sub(r'[^a-zA-Z\s]', '', tokens).split()  # Basic tokenization
    # Convert tokens to IDs using the vocabulary
    token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    # Pad or truncate to max_len
    if len(token_ids) < max_len:
        token_ids += [vocab['<PAD>']] * (max_len - len(token_ids))  # Padding
    else:
        token_ids = token_ids[:max_len] 
    return torch.tensor([token_ids]).to(device)  

# Predict the sentiment of a review
def predict_sentiment(review_text):
    # Preprocess the review
    input_ids = preprocess_review(review_text, vocab, max_len)
    
    # Make prediction
    with torch.no_grad():
        logits = model(input_ids)
        probabilities = torch.softmax(logits, dim=1)
        _, prediction = torch.max(probabilities, dim=1)
    
    return int(prediction), probabilities

# Main block
if __name__ == '__main__':
        
    # Load data
    df = pd.read_csv('IMDB Dataset.csv')
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    df['review'] = df['review'].apply(clean_text)

    # Sample 25 positive and 25 negative reviews
    positive_reviews = df[df['sentiment'] == 1].sample(25, random_state=42)  # Sample 25 positive reviews
    negative_reviews = df[df['sentiment'] == 0].sample(25, random_state=42)  # Sample 25 negative reviews

    # Concatenate positive and negative reviews to create a balanced dataset of 50 reviews
    df = pd.concat([positive_reviews, negative_reviews]).sample(frac=1, random_state=42)

    # - vocab: the vocabulary used for tokenizing text
    # - max_len: the maximum length of the tokenized review
    # - class_names: list of class names (e.g., ['negative', 'positive'])
    # - model: the trained Transformer model

    model = TransformerForSequenceClassification(config).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model = model.eval()

    # Streamlit app UI
    st.title('Movie Review Sentiment Analysis')
    st.write('Select a movie review from the list below to predict its sentiment.')

    # Selectbox to choose a review from df['review']
    selected_review = st.selectbox('Select a movie review', df['review'].values)

    # Predict button
    if st.button('Predict Sentiment'):
        if selected_review:
            # Make a prediction
            prediction, probabilities = predict_sentiment(selected_review)
            #prediction, probabilities = predict_sentiment(df.review[38388])

            # Get the class label
            class_names = ['Negative', 'Positive']
            sentiment = class_names[prediction]

            # Display the prediction result
            st.write(f"**Selected Review:** {selected_review}")
            st.write(f"**Predicted Sentiment:** {sentiment}")
            st.write(f"**Confidence:** {probabilities[0][prediction].item() * 100:.2f}%")


#        streamlit run streamlit_app.py

