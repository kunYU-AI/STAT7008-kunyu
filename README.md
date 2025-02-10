# Kunyu part: SOTA NLP models (SMaLL-100, VADER, IndoBERT)

## Prepare
We should input dataset first：
```
git clone https://github.com/IndoNLP/nusax.git
```

## Model: SMaLL-100
1. Training config:
Change the finetune/utils/train_config.py for training setting     
2. Tokenizer:
It is the special tokenizer downloaded from https://huggingface.co/alirezamsh/small100/blob/main/tokenization_small100.py     
3. Run finetune codes:
```
cd STAT7008-kunyu
python finetune/finetune_smallM2M.py
```

4. Ouputs:
models dir = './models/smallM2M100/'
checkpoints dir = './checkpts/checkpoint-{epochs number}'
5. evaluation:     
run
```python translate_m2m.py```

## Model: VADER
adding additional indonesian lexicon dataset
```
git clone https://github.com/fajri91/InSet.git 
```
then run 
```
python vader.py
```
output a vader confussion matrix png named as eval_vader.png

## Model: IndoBERT
1. Training config:
Change the finetune/ft_indobert.yaml for training setting
2. Run finetune codes:
```
python finetune/finetune_indobert.py
```
get the loss curve saved as loss_indobert.png

3. Outputs
checkpoints dir = './checkpts/indobert_model.pth'    
4. evaluation
run
```python eval_indobert.py``` 
and get eval_indobert.png of confusion matrix
