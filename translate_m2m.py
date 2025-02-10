import sys

import torch
from transformers import AutoModelForSeq2SeqLM
from finetune.Tokenizers.tokenization_small100 import SMALL100Tokenizer

# OUTPUT_MODEL_DIR = './checkpts/checkpoint-800'
OUTPUT_MODEL_DIR = './models/smallM2M100/'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Translation function
def translate_text(text, model, tokenizer):
    input_text = text['text']
    tokenizer.src_lang = 'id'
    tokenizer.tgt_lang = text['tgt_lang_code']
    model.to(DEVICE)
    model.eval()

    encoded_input = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=encoded_input['input_ids'],
            attention_mask=encoded_input['attention_mask'],
            max_length=128,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

if __name__ == "__main__":
    tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100")
    model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_MODEL_DIR)
    
        # translation test  
    test_sentences = [
        {
            "text": "Nikmati cicilan 0% hingga 12 bulan untuk pemesanan tiket pesawat air asia dengan kartu kredit bni!",
            "src_lang_code": "id",
            "tgt_lang_code": "en"
        },
        {
            "text": "Dekat dengan hotel saya menginap, hanya ditempuh jalan kaki, di sini banyak sekali pilihan makanannya, tempat yang luas, dan menyenangkan",
            "src_lang_code": "id",
            "tgt_lang_code": "jv"
        }
    ]

    for test in test_sentences:
        translation = translate_text(test, model, tokenizer)
        print(f"\nSource ({test['src_lang_code']}): {test['text']}")
        print(f"Target ({test['tgt_lang_code']}): {translation}")

# indonesian: "Dekat dengan hotel saya menginap, hanya ditempuh jalan kaki, di sini banyak sekali pilihan makanannya, tempat yang luas, dan menyenangkan"
# english: "Near the hotel I stayed in, reachable by foor, so many food choice here, the place is huge, and fun"
# javaness: "Cepak saka hotelku nginep, namung digawa mlaku, ing kene akeh tenan pilian panganane, panggonane sing amba, lan nyenengake"