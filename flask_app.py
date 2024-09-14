#%%
from flask import Flask, request, render_template,Blueprint
import torch
import torch.nn.functional as F
import pandas as pd
from utils_v4 import *
site = Blueprint('site', __name__, template_folder='templates')

with open('vocab_v4.pkl', 'rb') as f:
    vocab = pickle.load(f)

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


#%%
# Initialize the Flask app
app = Flask(__name__)

# Load the dataset and preprocess it
df = pd.read_csv('IMDB Dataset.csv')
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
df['review'] = df['review'].apply(clean_text)

# Sample 25 positive and 25 negative reviews
positive_reviews = df[df['sentiment'] == 1].sample(25, random_state=42)
negative_reviews = df[df['sentiment'] == 0].sample(25, random_state=42)

# Concatenate positive and negative reviews to create a balanced dataset of 50 reviews
df = pd.concat([positive_reviews, negative_reviews]).sample(frac=1, random_state=42)

model = TransformerForSequenceClassification(config).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
model = model.eval()



# - vocab: the vocabulary used for tokenizing text
# - max_len: the maximum length of the tokenized review
# - class_names: list of class names (e.g., ['negative', 'positive'])
# - model: the trained Transformer model

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


# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    selected_review = None
    sentiment = None
    confidence = None
    
    # If the form is submitted
    if request.method == 'POST':
        selected_review = request.form['review']
        
        # Predict sentiment
        prediction, probabilities = predict_sentiment(selected_review)
        sentiment = class_names[prediction]
        confidence = probabilities[0][prediction].item() * 100
    
    return render_template('index.html', reviews=df['review'].values, selected_review=selected_review, sentiment=sentiment, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)

# %%
#             python flask_app.py
