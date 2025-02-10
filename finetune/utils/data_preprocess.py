# data_preprocess.py
import pandas as pd
from datasets import Dataset
from torch.utils.data import Dataset as torDataset
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TranslationDataset:
    def __init__(self, file_name, tokenizer, model_id='t5', tgt_lang_code=None):
        self.file_name = file_name
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.tgt_lang_code = tgt_lang_code

    def load_data(self):
        """Load datasets from the specified CSV file."""
        df = pd.read_csv(self.file_name)[['indonesian', 'english', 'javanese']]
        return df

    def transform_data(self, df):
        """Transform raw DataFrame into training format."""
        transformed_data = []
        for _, row in df.iterrows():
            transformed_data.append({
                'src_text': row['indonesian'],
                'tgt_text': row['english'],
                'src_lang': 'indonesian',
                'tgt_lang': 'english'
            })
            transformed_data.append({
                'src_text': row['indonesian'],
                'tgt_text': row['javanese'],
                'src_lang': 'indonesian',
                'tgt_lang': 'javanese'
            })
        return pd.DataFrame(transformed_data)

    def preprocess_m2m(self, examples):
        """Tokenization for M2M model."""
        self.tokenizer.tgt_lang = self.tgt_lang_code
        self.tokenizer.src_lang = 'id'  # Source language is always Indonesian
        
        model_inputs = self.tokenizer(
            examples['src_text'],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(
                examples['tgt_text'],
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

        model_inputs['labels'] = targets['input_ids']
        return model_inputs

    def preprocess_t5(self, examples):
        """Tokenization for T5 model."""
        prefix = [f"translate {src} to {tgt}: " for src, tgt in zip(examples['src_lang'], examples['tgt_lang'])]
        inputs = [f"{p}{txt}" for p, txt in zip(prefix, examples['src_text'])]
        
        model_inputs = self.tokenizer(
            inputs,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        targets = self.tokenizer(
            examples['tgt_text'],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        model_inputs['labels'] = targets['input_ids']
        return model_inputs
    
    def preprocess(self):
        """Preprocess examples based on the model type."""
        df = self.load_data()
        if self.model_id == 't5':
            transformed_df = self.transform_data(df)
            dataset = Dataset.from_pandas(transformed_df)
            ready_data = dataset.map(self.preprocess_t5, batched=True)
        
        elif self.model_id == 'm2m':
            if self.tgt_lang_code is None:
                raise ValueError("Target language code must be provided for M2M model.")
            elif self.tgt_lang_code == 'en':
                df = df[['indonesian', 'english']]
            elif self.tgt_lang_code == 'jv':
                df = df[['indonesian', 'javanese']]
            
            df.columns = ['src_text', 'tgt_text']
            dataset = Dataset.from_pandas(df)
            ready_data = dataset.map(self.preprocess_m2m, batched=True)
        
        return ready_data

# Example usage:
# tokenizer = ...  # Initialize your tokenizer here
# translation_dataset = TranslationDataset(file_name='train.csv', tokenizer=tokenizer, model_id='t5', tgt_lang_code=None)

class SentimentDataset(torDataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
