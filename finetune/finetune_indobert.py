import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, classification_report
import yaml
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt

from utils.data_preprocess import SentimentDataset
from utils.train import training_indobert 


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

class SentimentClassifier(nn.Module):
    def __init__(self, model_name, num_classes=3):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[0][:, 0, :]  # CLS token
        output = self.drop(pooled_output)
        return self.fc(output)

def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    losses = []
    correct_predictions = 0
    total_predictions = 0
    
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            
            losses.append(loss.item())
            
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += len(labels)
            
            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
    
    return (
        np.mean(losses), 
        correct_predictions.double() / total_predictions,
        predictions,
        actual_labels
    )

def main():
    # load config
    config = load_config('./finetune/ft_indobert.yaml')
    print(f"config: {config}\n")
    file_pth = config['data_path']
    model_config = config['model']
    train_config = config['train']
    output_config = config['output']
    
    MODEL_NAME = model_config['NAME']
    MAX_LEN = model_config['MAX_LEN']
    TRAIN_BATCH_SIZE = train_config['TRAIN_BATCH_SIZE']
    VALID_BATCH_SIZE = train_config['VALID_BATCH_SIZE']
    TEST_BATCH_SIZE = train_config['TEST_BATCH_SIZE']
    EPOCHS = train_config['EPOCHS']
    LEARNING_RATE = float(train_config['LEARNING_RATE'])
    
    CHECKPOINTS = output_config['CHECKPOINTS']
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load data
    train_df = pd.read_csv(file_pth+'train.csv')
    valid_df = pd.read_csv(file_pth+'valid.csv')
    test_df = pd.read_csv(file_pth+'test.csv')
    
    # map sentiment labels with number 
    label_dict = {'negative': 0, 'neutral': 1, 'positive': 2}
    train_df['label'] = train_df['label'].map(label_dict)
    valid_df['label'] = valid_df['label'].map(label_dict)
    test_df['label'] = test_df['label'].map(label_dict)
    
    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # create dataset
    train_dataset = SentimentDataset(
        texts=train_df['text'].values,
        labels=train_df['label'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    valid_dataset = SentimentDataset(
        texts=valid_df['text'].values,
        labels=valid_df['label'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    test_dataset = SentimentDataset(
        texts=test_df['text'].values,
        labels=test_df['label'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    # create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False
    )
    
    # initialize model
    model = SentimentClassifier(MODEL_NAME)
    model = model.to(DEVICE)
    
    # define loss and optimizer
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # train
    best_accuracy = 0
    train_losses = []
    val_losses = []
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        
        train_loss, train_acc = training_indobert(
            model,
            train_loader,
            loss_fn,
            optimizer,
            DEVICE
        )
        
        print(f'Train loss {train_loss} accuracy {train_acc}')
        train_losses.append(train_loss)
        
        # valuation
        val_loss, val_acc, predictions, actual = evaluate(
            model,
            valid_loader,
            loss_fn,
            DEVICE
        )
        
        print(f'Val loss {val_loss} accuracy {val_acc}')
        val_losses.append(val_loss)
        
        print("\nValidation Classification Report:")
        print(classification_report(actual, predictions))
        
        # store the best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), CHECKPOINTS)
    
    plt.figure(figsize=(10,6))
    plt.plot(range(1, EPOCHS+1), train_losses, label='Train loss', marker='o')
    plt.plot(range(1, EPOCHS+1), val_losses, label='Valid loss', marker='o')
    
    plt.title('Finetune: Train and Validation Loss over Epochs', fontsize=24)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.xticks(range(1, EPOCHS + 1))  
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend() 
    plt.grid() 
    plt.savefig("loss_indobert.png", dpi=300)

    # prediction on test
    model.load_state_dict(torch.load(CHECKPOINTS))
    test_loss, test_acc, test_predictions, test_actual = evaluate(
        model,
        test_loader,
        loss_fn,
        DEVICE
    )
    
    print("\nTest Results:")
    print(f'Test loss {test_loss} accuracy {test_acc}')
    print("\nTest Classification Report:")
    print(classification_report(test_actual, test_predictions))

if __name__ == "__main__":
    main()