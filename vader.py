# !git clone https://github.com/IndoNLP/nusax.git
# !git clone https://github.com/fajri91/InSet.git    # 补充印度尼西亚的情感辞典

import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from textblob import TextBlob

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

import os

def load_sentiment_dict(pos_file, neg_file):
    positive_words = {}
    negative_words = {}
    
    # 加载并标准化权重
    def normalize_weight(weight, max_weight=5.0):
        return (float(weight) / max_weight) * 2 - 1  # 转换到 [-1, 1] 范围
    
    with open(pos_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            word, weight = line.strip().split('\t')
            positive_words[word] = normalize_weight(weight)
    
    with open(neg_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            word, weight = line.strip().split('\t')
            negative_words[word] = -normalize_weight(weight)  # 负面词转为负值
            
    return positive_words, negative_words

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s!?.]', '', text)
    text = re.sub(r'(.)\1+', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = word_tokenize(text)
    stop_words = set(stopwords.words('indonesian'))
    negation_words = {'tidak', 'bukan', 'tak', 'belum'}
    stop_words = stop_words - negation_words
    
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    
    return ' '.join(words)

class CustomSentimentAnalyzer:
    def __init__(self, positive_words, negative_words):
        self.positive_words = positive_words
        self.negative_words = negative_words
        self.sia = SentimentIntensityAnalyzer()
        
    def get_ngrams(self, text, n=2):
        words = text.split()
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    
    def check_negation(self, text):
        negation_words = {'tidak', 'bukan', 'tak', 'belum'}
        words = text.split()
        return any(word in negation_words for word in words)
    
    def analyze_sentiment(self, text):
        processed_text = preprocess(text)
        words = processed_text.split()
        # vader scores
        vader_scores = self.sia.polarity_scores(text)
        
        # extra lexicon scores
        word_scores = []
        for word in words:
            score = self.positive_words.get(word, 0) + self.negative_words.get(word, 0)
            word_scores.append(score)
            
        # N-gram
        bigrams = self.get_ngrams(processed_text, 2)
        bigram_scores = []
        for bigram in bigrams:
            if bigram in self.positive_words:
                bigram_scores.append(self.positive_words[bigram])
            elif bigram in self.negative_words:
                bigram_scores.append(self.negative_words[bigram])
        
        # check the negative works
        has_negation = self.check_negation(processed_text)
        
        # textblob sentiment score
        blob_score = TextBlob(text).sentiment.polarity
        
        # final scores
        custom_score = np.mean(word_scores) if word_scores else 0
        bigram_score = np.mean(bigram_scores) if bigram_scores else 0
        
        vader_weight = 0.05
        custom_weight = 0.5
        bigram_weight = 0.35
        blob_weight = 0.1
        
        final_score = (vader_scores['compound'] * vader_weight +
                      custom_score * custom_weight +
                      bigram_score * bigram_weight +
                      blob_score * blob_weight)
        
        # check if it has negation
        if has_negation:
            final_score *= -1

        if final_score >= 0.05:
            return 'positive'
        elif final_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
def evaluate_model(y_true, y_pred):
    print("\nAccuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
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
    plt.savefig('eval_vader.png')
    plt.show()

def main():
    cwd = os.getcwd()
    file_pth = cwd+'/nusax/datasets/sentiment/indonesian/'
    
    train_df = pd.read_csv(file_pth+'train.csv')
    valid_df = pd.read_csv(file_pth+'valid.csv')
    test_df = pd.read_csv(file_pth+'test.csv')
    
    ext_data_pth = cwd+'/InSet/'
    positive_words, negative_words = load_sentiment_dict(ext_data_pth+'positive.tsv', 
                                                       ext_data_pth+'negative.tsv')
    
    analyzer = CustomSentimentAnalyzer(positive_words, negative_words)
    
    print("Evaluating on train set:")
    train_predictions = train_df['text'].apply(analyzer.analyze_sentiment).tolist()
    evaluate_model(train_df['label'], train_predictions)

    print("Evaluating on validation set:")
    valid_predictions = valid_df['text'].apply(analyzer.analyze_sentiment).tolist()
    evaluate_model(valid_df['label'], valid_predictions)
    
    print("\nEvaluating on test set:")
    test_predictions = test_df['text'].apply(analyzer.analyze_sentiment).tolist()
    evaluate_model(test_df['label'], test_predictions)

if __name__ == '__main__':
    main()