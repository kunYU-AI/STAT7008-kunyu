class TrainConfig:
    def __init__(self):
        # target translation language
        self.tgt_lang_code = 'en'
            
        # trainer arguments
        self.batch_size = 16
        self.learning_rate = 5e-5
        self.num_epochs = 1
        self.max_length = 128
        self.weight_decay = 0.01
        self.gradient_accumulation_steps = 8
        
        # file path, current work dir=STAT7008_kunyu_SOTA
        self.output_model_dir = './models/smallM2M100/'
        self.input_file_dir = '../nusax/datasets/mt/'    # since we run python finetune/finetune_smallM2M100.py