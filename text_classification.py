from utils import *


# Load IMDb Dataset
df = pd.read_csv('IMDB Dataset.csv')
print('IMDB Dataset.csv data loaded ...')
# Preprocess the dataset
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)  # Convert sentiments to binary

# Simple tokenization process
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    tokens = text.split()
    return tokens

# Build Vocabulary
def build_vocab(reviews):
    vocab = Counter()
    for review in reviews:
        tokens = tokenize(review)
        vocab.update(tokens)
    
    vocab = {word: i+3 for i, (word, _) in enumerate(vocab.most_common())}  # Start index from 3
    vocab['<PAD>'] = 0  # Padding token
    vocab['<UNK>'] = 1  # Unknown token
    vocab['[CLS]'] = 2  # [CLS] token
    return vocab

# Convert text to numerical indices
def text_to_indices(text, vocab, max_len):
    tokens = ['[CLS]'] + tokenize(text)
    indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    if len(indices) < max_len:
        indices += [vocab['<PAD>']] * (max_len - len(indices))  # Padding
    else:
        indices = indices[:max_len]  # Truncate if longer than max_len
    return indices

# Maximum sequence length (adjusted to include [CLS] token)
max_len = 401  # max_len (400) + 1 for [CLS] token

# Tokenize and encode the dataset
if os.path.exists('tokenized_reviews_vocab_v4.npy') and os.path.exists('vocab_v4.pkl'):
    tokenized_reviews = np.load('tokenized_reviews_vocab_v4.npy')
    with open('vocab_v4.pkl', 'rb') as f:
        vocab = pickle.load(f)
else:
    reviews = df['review'].tolist()
    vocab = build_vocab(reviews)
    # Convert reviews to input IDs
    tokenized_reviews = [text_to_indices(review, vocab, max_len) for review in reviews]
    np.save('tokenized_reviews_vocab_v4.npy', np.array(tokenized_reviews))
    with open('vocab_v4.pkl', 'wb') as f:
        pickle.dump(vocab, f)

# Convert arrays into PyTorch tensors
inputs_input_ids = torch.tensor(tokenized_reviews).to(device)
labels = torch.tensor(df['sentiment'].values).to(device)

print('Tokenization and vocabulary buidling complete ...')

# Split the dataset into training, validation, and test sets (70%, 15%, 15%)
train_inputs, valid_test_inputs, train_labels, valid_test_labels = train_test_split(
    inputs_input_ids, labels, test_size=0.3, random_state=100, shuffle=True
)
valid_inputs, test_inputs, valid_labels, test_labels = train_test_split(
    valid_test_inputs, valid_test_labels, test_size=0.5, random_state=100, shuffle=True
)

# Create DataLoader
train_dataset = TensorDataset(train_inputs, train_labels)
valid_dataset = TensorDataset(valid_inputs, valid_labels)
test_dataset = TensorDataset(test_inputs, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

print('Data loaders complete ...')

# Training function

def train(model, train_loader, valid_loader, optimizer, scheduler, loss_fn, device, epochs, early_stopping_patience=5):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_accuracy = 0  # Track the best validation accuracy
    patience_counter = 0  # Counter for early stopping
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Evaluate on the validation set
        val_loss, val_acc = evaluate(model, valid_loader, loss_fn, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        scheduler.step()

        # Early stopping
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_.pth')
            
        #     print(f"Epoch {epoch+1}: Best model saved with accuracy: {best_accuracy:.2f}%")
        # else:
        #     patience_counter += 1
        #     if patience_counter >= early_stopping_patience:
        #         print("Early stopping triggered.")
        #         break
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
        
    return train_losses, val_losses, train_accuracies, val_accuracies

# Evaluation function
def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# Plotting loss and accuracy
def plot_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('Train_valid_curves_v4.png')
    plt.show()
    plt.close()

def hyperparameter_search(hyperparameter_combinations, results_df):
    for i, params in enumerate(hyperparameter_combinations):
        print(f"\nRunning hyperparameter set {i+1}/{len(hyperparameter_combinations)}")
        
        # Unpack hyperparameters
        lr, num_hidden_layers, num_attention_heads, hidden_size, intermediate_size, hidden_dropout_prob, activation_function = params
        
        # Prepare DataLoader with current batch_size
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=32)
        
        # Initialize the model with current hyperparameters
        config = TransformerConfig(
            vocab_size=len(vocab),
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            max_position_embeddings=max_len,
            num_labels=2,
            activation_function=activation_function
        )
        model = TransformerForSequenceClassification(config).to(device)
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        loss_fn = nn.CrossEntropyLoss()
        
        # Train the model
        epochs = 100 
        
        train_losses, val_losses, train_accuracies, val_accuracies = train(model, train_loader, valid_loader, optimizer, scheduler, loss_fn, device, epochs, early_stopping_patience=5)
        
        # Get the best validation accuracy
        best_val_accuracy = max(val_accuracies)
        best_val_loss = min(val_losses)

        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        
        # Save the results
        current_result = pd.DataFrame([{
            'lr': lr,
            'num_hidden_layers': num_hidden_layers,
            'num_attention_heads': num_attention_heads,
            'hidden_size': hidden_size,
            'intermediate_size': intermediate_size,
            'hidden_dropout_prob': hidden_dropout_prob,
            'activation_function': activation_function,
            'validation_accuracy': best_val_accuracy,
            'test_accuracy': test_acc
         }])

        results_df = pd.concat([results_df, current_result], ignore_index=True)
        
        # save the model if it's the best so far
        if best_val_accuracy == results_df['validation_accuracy'].max():
            torch.save(model.state_dict(), 'best_model.pth')
            results_df.to_csv('results_df_v4_upd.csv', index=False)
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.2f}%")

            # Plot loss and accuracy curves
            plot_curves(train_losses, val_losses, train_accuracies, val_accuracies)
            # Load the best model and evaluate on the test set
            #model.load_state_dict(torch.load('best_model.pth'))
            test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

        
    return results_df

# Load the best saved model
# model = TransformerForSequenceClassification(config)
# model.load_state_dict(torch.load('best_model.pth'))
# model.to(device)


def main():
    # DataFrame to store hyperparameters and validation scores:
    results_df = pd.DataFrame(columns=[
        'lr',
        'num_hidden_layers',
        'num_attention_heads',
        'hidden_size',
        'intermediate_size',
        'hidden_dropout_prob',
        'activation_function',
        'validation_accuracy',
        'test_accuracy'
    ])
    results_df = hyperparameter_search(sampled_combinations, results_df)
    # Sort the results by validation accuracy
    sorted_results = results_df.sort_values(by='validation_accuracy', ascending=False)

    # Display the top 5 configurations
    print("Top 5 Hyperparameter Configurations:")
    print(sorted_results.head(5))

        

if __name__ == '__main__':
    main()







