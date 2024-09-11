import pandas as pd
import numpy as np
import re, os, random, torch
import seaborn as sns
from collections import defaultdict
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

#For Reproducibility
def reproducibility_requirements(seed=100):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

reproducibility_requirements()

# Function to clean the review text
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers (optional, since BERT can handle this)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert text to lowercase
    text = text.lower().strip()
    return text



# Custom dataset class inheriting from torch.utils.data.Dataset.
# This class is used to handle the movie reviews and prepare them in a format that can be directly fed into the BERT model.
class MovieDataset(Dataset):

    # Initialization method to pass the data (reviews and targets), tokenizer, and max sequence length.
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    # Returns the total number of samples in the dataset.
    def __len__(self):
        return len(self.reviews)

    # Retrieves a single data item at the specified index.
    def __getitem__(self, item):
        # Get the review text and target sentiment label for the given item.
        review = str(self.reviews[item])
        target = self.targets[item]

        # Tokenize the review text using the BERT tokenizer.
        # The 'encode_plus' method handles tokenization, adding special tokens (e.g., [CLS], [SEP]),
        # padding to the specified maximum length (max_len), and generating the attention mask.
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,  # Adds [CLS] and [SEP] tokens automatically.
            max_length=self.max_len,  # Truncate or pad the sequence to the maximum length of 400 tokens.
            return_token_type_ids=False,  # We don't need token type IDs for single-sentence tasks.
            pad_to_max_length=True,  # Pads the sequences to max_length with zeros.
            return_attention_mask=True,  # Returns the attention mask (1 for tokens, 0 for padding).
            return_tensors='pt',  # Return PyTorch tensors (input to the BERT model).
            truncation=True  # Truncate sequences longer than max_length.
        )

        # Return a dictionary containing the review text, input_ids, attention mask, and target label.
        return {
            'review_text': review,  # The original review text (useful for debugging).
            'input_ids': encoding['input_ids'].flatten(),  # Flatten the tensor to a 1D tensor of token IDs.
            'attention_mask': encoding['attention_mask'].flatten(),  # Flatten the tensor to a 1D tensor.
            'targets': torch.tensor(target, dtype=torch.long)  # Target label as a long tensor (for classification).
        }

# Function to create a data loader for batching data during training or evaluation.
# Takes a DataFrame, tokenizer, maximum sequence length, and batch size as input.
def create_data_loader(df, tokenizer, max_len, batch_size):
    # Instantiate the custom MovieDataset with the reviews, targets (sentiments), tokenizer, and max_len.
    ds = MovieDataset(
        reviews=df.review.to_numpy(),  # Convert the 'review' column to a NumPy array for indexing efficiency.
        targets=df.sentiment.to_numpy(),  # Convert the 'sentiment' column (labels) to a NumPy array.
        tokenizer=tokenizer,  # The BERT tokenizer for tokenizing the text.
        max_len=max_len  # Maximum sequence length for each tokenized review.
    )

    # Return a DataLoader object that will handle batching and shuffling of the dataset during training or evaluation.
    return DataLoader(
        ds,  # Dataset instance (MovieDataset).
        batch_size=batch_size,  # Number of samples in each batch (32 reviews per batch here).
        num_workers=4  # Number of subprocesses to use for data loading (parallelizes data fetching for speed).
    )



# Define a custom PyTorch model for sentiment classification, subclassing nn.Module.
class TextClassifier(nn.Module):

    # Initialize the model. The constructor takes the number of classes as input (binary classification here).
    def __init__(self, n_classes):
        super(TextClassifier, self).__init__()
        
        # Load the pre-trained BERT model (bert-base-cased) from Hugging Face's transformer library.
        # return_dict=False ensures backward compatibility, so BERT outputs tuple instead of dictionary.
        self.bert = BertModel.from_pretrained('bert-base-cased', return_dict=False)
        
        # Dropout layer to prevent overfitting. The dropout probability is set to 30% (a common value).
        self.drop = nn.Dropout(p=0.3)
        
        # The final fully connected layer which outputs 'n_classes' logits (in this case, 2: positive or negative).
        # The size of the input is 'self.bert.config.hidden_size', which is 768 for the base BERT model.
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    # Forward method to define how the data will pass through the model during training/inference.
    def forward(self, input_ids, attention_mask):
        # Pass input IDs and attention mask to the BERT model. BERT returns two outputs: 
        # 1. Sequence output (token embeddings for each token in the input), 
        # 2. Pooled output (embedding of the [CLS] token representing the entire sentence).
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Apply dropout to the pooled output (representing the review as a whole).
        output = self.drop(pooled_output)
        
        # Pass the pooled output through the final fully connected layer to obtain class logits.
        return self.out(output)

# Function to save the model after each epoch
def save_model(model, epoch, optimizer, scheduler, save_dir="model_checkpoints"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model_save_path = f"{save_dir}/model_epoch_{epoch}.pth"
    optimizer_save_path = f"{save_dir}/optimizer_epoch_{epoch}.pth"
    scheduler_save_path = f"{save_dir}/scheduler_epoch_{epoch}.pth"
    
    # Save model state dictionary (weights)
    torch.save(model.state_dict(), model_save_path)
    
    # Optionally save optimizer and scheduler states
    torch.save(optimizer.state_dict(), optimizer_save_path)
    torch.save(scheduler.state_dict(), scheduler_save_path)
    
    print(f"Model and optimizer saved to {save_dir} for epoch {epoch}")


# Function to train the model for one epoch and return accuracy and average loss
def train_epoch(
    model,
    data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    n_examples,
   
):
    model = model.train()  # Set the model to training mode

    losses = []
    correct_predictions = 0

    # Loop through the batches of data in the data loader
    for d in data_loader:
        input_ids = d["input_ids"].to(device)  # Move input_ids to GPU or CPU (depending on availability)
        attention_mask = d["attention_mask"].to(device)  # Move attention_mask to GPU or CPU
        targets = d["targets"].to(device)  # Move target labels to GPU or CPU

        # Forward pass through the model
        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        
        # Get the predicted class labels
        _, preds = torch.max(outputs, dim=1)
        
        # Calculate the loss
        loss = loss_fn(outputs, targets)

        # Count the correct predictions for accuracy calculation
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        # Backward pass for gradient calculation
        loss.backward()

        # Clip gradients to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update the model parameters
        optimizer.step()

        # Update the learning rate based on the scheduler
        scheduler.step()

        # Reset gradients for the next batch
        optimizer.zero_grad()

    # Calculate accuracy and average loss for the epoch
    accuracy = correct_predictions.double() / n_examples
    avg_loss = np.mean(losses)
  
    return accuracy, avg_loss

# Function to evaluate the model on validation or test data.
# Takes the model, data loader, loss function, device, and the number of examples in the dataset.
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()  # Set the model to evaluation mode. This disables dropout and gradient tracking.
    
    losses = []  # List to store loss values for each batch.
    correct_predictions = 0  # Counter to store the number of correct predictions.

    # Disable gradient calculations for evaluation as it saves memory and computational resources.
    with torch.no_grad():
        # Iterate over batches from the data loader.
        for d in data_loader:
            # Move input data to the appropriate device (GPU/CPU).
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            # Forward pass: Get the model's predictions without backpropagation.
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Extract the predicted class (0 or 1) by finding the index of the maximum logit for each example.
            _, preds = torch.max(outputs, dim=1)
            
            # Compute the loss for the current batch.
            loss = loss_fn(outputs, targets)

            # Increment the count of correct predictions by comparing model predictions to the targets.
            correct_predictions += torch.sum(preds == targets)
            
            # Append the loss for this batch to the losses list.
            losses.append(loss.item())

    # Calculate the accuracy as the ratio of correct predictions to the total number of examples.
    accuracy = correct_predictions.double() / n_examples
    
    # Return the average accuracy and the mean loss over all batches.
    return accuracy, np.mean(losses)


# Function to get predictions from the model for the provided data loader (test/validation set).
def get_predictions(model, data_loader):
    model = model.eval()  # Set the model to evaluation mode (no dropout, no gradient updates).

    review_texts = []  # List to store the actual review texts for later inspection.
    predictions = []  # List to store predicted labels (classes).
    prediction_probs = []  # List to store predicted probabilities for each class.
    real_values = []  # List to store actual target labels.

    # Disable gradient calculations during inference to save memory and speed up computations.
    with torch.no_grad():
        # Loop over each batch of data in the data loader.
        for d in data_loader:

            texts = d["review_text"]  # Extract review texts from the batch.
            input_ids = d["input_ids"].to(device)  # Move input_ids to the appropriate device (GPU/CPU).
            attention_mask = d["attention_mask"].to(device)  # Move attention_mask to the device.
            targets = d["targets"].to(device)  # Move target labels to the device.

            # Forward pass through the model to get raw logits for each class.
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Get predicted class by taking the index of the maximum logit value.
            _, preds = torch.max(outputs, dim=1)

            # Apply softmax to convert logits into probabilities.
            probs = F.softmax(outputs, dim=1)

            # Store the review texts, predictions, probabilities, and real labels.
            review_texts.extend(texts)  # Append the actual review texts for this batch.
            predictions.extend(preds)  # Append the predicted labels.
            prediction_probs.extend(probs)  # Append the predicted probabilities.
            real_values.extend(targets)  # Append the real target labels.

    # Convert predictions, probabilities, and real values to PyTorch tensors and move them to CPU.
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()

    # Return the review texts, predicted labels, predicted probabilities, and actual labels.
    return review_texts, predictions, prediction_probs, real_values


# Function to display the confusion matrix.
def show_confusion_matrix(confusion_matrix, class_names):
    plt.figure(figsize=(8, 6))
    
    # Create a heatmap from the confusion matrix.
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
                       xticklabels=class_names, yticklabels=class_names)

    # Rotate the tick labels for better readability.
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    
    # Set the labels and title.
    plt.ylabel('True Sentiment', fontsize=12)
    plt.xlabel('Predicted Sentiment', fontsize=12)
    plt.title('Confusion Matrix', fontsize=15)
    plt.savefig('/classification_report.png')
    plt.close()