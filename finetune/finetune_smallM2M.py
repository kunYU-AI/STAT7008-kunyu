from transformers import AutoModelForSeq2SeqLM
from utils.data_preprocess import TranslationDataset
from utils.train import training_m2m
from Tokenizers.tokenization_small100 import SMALL100Tokenizer
from utils.train_config import TrainConfig

import warnings
warnings.filterwarnings('ignore')

def load_model(model_name, new_token_emb):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.resize_token_embeddings(new_token_emb)
    return model

if __name__ == '__main__':
    config = TrainConfig()
    tgt_lang_code = config.tgt_lang_code
    output_model_dir = config.output_model_dir
    input_file_dir = config.input_file_dir
    
    tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100")
    # Load data
    trans_train_dataset = TranslationDataset(file_name=input_file_dir+'train.csv', 
                                             tokenizer=tokenizer, 
                                             model_id='m2m',
                                             tgt_lang_code=tgt_lang_code)
    trans_valid_dataset = TranslationDataset(file_name=input_file_dir+'valid.csv', 
                                             tokenizer=tokenizer, 
                                             model_id='m2m', 
                                             tgt_lang_code=tgt_lang_code)

    tokenized_train = trans_train_dataset.preprocess()
    tokenized_valid = trans_valid_dataset.preprocess()
    
    # load model
    input_model_dir = "alirezamsh/small100"    # initially, without trained model
    # input_model_dir = output_model_dir#
    model = load_model(input_model_dir, len(tokenizer))

    trainer = training_m2m(model, tokenizer, tokenized_train, tokenized_valid, f'checkpts')

    trainer.train()

    trainer.save_model(output_model_dir)
