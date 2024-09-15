# Movie Review Sentiment Analysis based on Transformer model

This repository contains a **Streamlit app** for sentiment analysis on movie reviews based on a Transformer model. The app predicts whether a given review has a **positive** or **negative** sentiment based on the IMDb movie reviews dataset.

## Features
- **Interactive Sentiment Prediction**: Users can  select a pre-existing review from the dataset, and the app will predict whether the sentiment is positive or negative. Here is the [Streamlit Deployment app](https://textclassificationdemo.streamlit.app/).
- **Confidence Score**: The app also displays the confidence score of the prediction, indicating how confident the model is in its prediction.

## Usage of the [streamlit app](https://textclassificationdemo.streamlit.app/)

1. Open the Streamlit app in your web browser by navigating to the URL provided in the command output.
2. Select a pre-existing review from the dropdown menu.
3. Click the "Predict" button to get the sentiment prediction.
4. The app will display the predicted sentiment (positive or negative) and the confidence score.

NB: Importantly the whole workflow is outlined and clearly explained in the following [Notebook](https://stevenndungu.github.io/text_classification)
 

## Local Setup and Installation

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

NB: **The Same can be run on Flask app:** [Flask App script](https://github.com/stevenndungu/text_classification/blob/main/flask_app.py) 



