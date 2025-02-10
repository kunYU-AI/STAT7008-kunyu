import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import yaml
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from finetune.utils.data_preprocess import SentimentDataset
from finetune.utils.train import training_indobert

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
    misclassified_samples = []  # 用于存储错误分类的样本
    k = 0
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
            
            # 记录错误分类的样本
            for i in range(len(labels)):
                k += 1
                if preds[i] != labels[i]:
                    misclassified_samples.append({
                        'id': k,  # 假设输入文本在batch中
                        'predicted': preds[i].item(),
                        'true': labels[i].item()
                    })

    return (
        np.mean(losses), 
        correct_predictions.double() / total_predictions,
        predictions,
        actual_labels,
        misclassified_samples  # 返回错误分类样本
    )

def main():
    # load config
    config = load_config('./finetune/ft_indobert.yaml')
    print(f"config: {config}\n")
    file_pth = config['data_path']
    model_config = config['model']
    output_config = config['output']
    train_config = config['train']
    
    TEST_BATCH_SIZE = train_config['TEST_BATCH_SIZE']
    MODEL_NAME = model_config['NAME']
    MAX_LEN = model_config['MAX_LEN']
    CHECKPOINTS = output_config['CHECKPOINTS']
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load data
    train_df = pd.read_csv(file_pth + 'train.csv')
    valid_df = pd.read_csv(file_pth + 'valid.csv')
    test_df = pd.read_csv(file_pth + 'test.csv')
    
    # map sentiment labels with number 
    label_dict = {'negative': 0, 'neutral': 1, 'positive': 2}
    test_df['label'] = test_df['label'].map(label_dict)
    
    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    test_dataset = SentimentDataset(
        texts=test_df['text'].values,
        labels=test_df['label'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False
    )
    
    # initialize model
    model = SentimentClassifier(MODEL_NAME)
    model = model.to(DEVICE)
    
    # define loss
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)
    
    # prediction on test
    model.load_state_dict(torch.load(CHECKPOINTS))
    test_loss, test_acc, test_predictions, test_actual, misclassified_samples = evaluate(
        model,
        test_loader,
        loss_fn,
        DEVICE
    )
    
    print("\nTest Results:")
    print(f'Test loss {test_loss} accuracy {test_acc}')
    print("\nTest Classification Report:")
    
    # 打印错误分类的结果
    print("\nMisclassified Samples:")
    for sample in misclassified_samples:
        print(f"Text: {sample['id']}, Predicted: {sample['predicted']}, True: {sample['true']}")

    # 混淆矩阵
    cm = confusion_matrix(test_actual, test_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                annot_kws={"size": 32},  # 设置注释字体大小
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title('Confusion Matrix', fontsize=24)  # 设置标题字体大小
    plt.ylabel('True Label', fontsize=18)  # 设置y轴标签字体大小
    plt.xlabel('Predicted Label', fontsize=18)  # 设置x轴标签字体大小
    plt.xticks(fontsize=20)  # 设置x轴标签字体大小
    plt.yticks(fontsize=20)  # 设置y轴标签字体大小
    plt.savefig('eval_indobert.png', dpi=300)
    
    print(classification_report(test_actual, test_predictions))

if __name__ == "__main__":
    main()