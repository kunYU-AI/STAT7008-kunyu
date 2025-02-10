import numpy as np
import torch
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from .train_config import TrainConfig
import datasets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def training_m2m(model, tokenizer, train_dataset, valid_dataset, output_dir):
    model.to(DEVICE)
    train_dataset
    valid_dataset
    
    config = TrainConfig()

    args = Seq2SeqTrainingArguments(
        output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        weight_decay=config.weight_decay,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        save_total_limit=1,
        num_train_epochs=config.num_epochs,
        predict_with_generate=True,
        logging_dir='logs',
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    metric = datasets.load_metric("sacrebleu",trust_remote_code=True)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        print(f"Decoded preds: {len(decoded_preds)}, Decoded labels: {len(decoded_labels)}")
        
        # Check if the number of predictions and references match
        if len(decoded_preds) != len(decoded_labels):
            raise ValueError(f"Number of predictions ({len(decoded_preds)}) does not match number of references ({len(decoded_labels)})")
            
        references = [[label] for label in decoded_labels]
        result = {}
        # Compute sacreBLEU
        bleu_result = metric.compute(predictions=decoded_preds, references=references)
        result["bleu"] = bleu_result["score"]
        
        
        with open('metric_results.txt', 'a') as f:  
            f.write(f"BLEU Score: {result['bleu']}\n")
            
        return result

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    return trainer

def training_indobert(model, data_loader, loss_fn, optimizer, device):
    model.train()
    losses = []
    correct_predictions = 0
    total_predictions = 0
    
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
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    return np.mean(losses), correct_predictions.double() / total_predictions