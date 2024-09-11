
#%%
# Import necessary libraries
import streamlit as st
from utils import *

df = pd.read_csv('IMDB Dataset.csv')
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x=='positive' else 0)
df['review'] = df['review'].apply(clean_text)

#Sample 25 positive and 25 negative

# Filter positive and negative sentiment reviews
positive_reviews = df[df['sentiment'] == 1].sample(25, random_state=42)  # Sample 25 positive reviews
negative_reviews = df[df['sentiment'] == 0].sample(25, random_state=42)  # Sample 25 negative reviews

# Concatenate the positive and negative reviews to create a balanced dataset of 50 reviews
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
        
        # Get the class label
        class_names = ['Negative', 'Positive']
        sentiment = class_names[prediction]
        
        # Display the prediction result
        st.write(f"**Selected Review:** {selected_review}")
        st.write(f"**Predicted Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {probabilities[0][prediction].item() * 100:.2f}%")

# %%
