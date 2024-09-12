#%%
from flask import Flask, request, render_template,Blueprint
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
import pandas as pd
from utils import clean_text, TextClassifier
site = Blueprint('site', __name__, template_folder='templates')
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

# Initialize the model and load the saved state
class_names = ['negative', 'positive']
model = TextClassifier(len(class_names))
model.load_state_dict(torch.load('best_model_state.bin', map_location=torch.device('cpu')))
model = model.eval()

# Function to predict sentiment
def predict_sentiment(review_text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    encoded_review = tokenizer.encode_plus(
        review_text,
        max_length=160,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    
    input_ids = encoded_review['input_ids']
    attention_mask = encoded_review['attention_mask']
    
    # Get the model's prediction
    outputs = model(input_ids, attention_mask)
    probs = F.softmax(outputs, dim=1)
    _, prediction = torch.max(probs, dim=1)
    
    return int(prediction), probs

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
