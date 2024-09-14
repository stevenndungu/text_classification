# Movie Review Sentiment Analysis with BERT

This repository contains a **Streamlit app** for sentiment analysis on movie reviews based on a Transformer model. The app predicts whether a given review has a **positive** or **negative** sentiment based on the IMDb movie reviews dataset.

## Features
- **Interactive Sentiment Prediction**: Users can input a movie review or select a pre-existing review from the dataset, and the app will predict whether the sentiment is positive or negative.
- **Streamlit Deployment**: The app is built with Streamlit, allowing for a simple and interactive UI.


## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/stevenndungu/text_classification.git
   
2. Change into the project directory:
	 ```bash
   cd your-repo-name
   
3. Create a virtual environment and activate it:
	```bash
	python -m venv venv
	source venv/bin/activate  # On Windows, use: venv\Scripts\activate
	
4. Install the required packages:
	```bash
	pip install -r requirements.txt
	
5. Make sure the pre-trained model (best_model_state.bin) is in the same directory as app.py. If not, download and place it there.

6. Run the Streamlit app :
	```bash
	streamlit run streamlit_app.py

