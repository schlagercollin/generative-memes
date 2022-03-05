"""
Some global hyperparameters
"""

# Batch size during training
batch_size = 128
workers = 8
img_size = 64

# captioning settings
caption_batch_size = 128
caption_num_epochs = 2000
caption_save_every = 20

# Refined language model default params
refined_model_vocab_embed_size      = 512
refined_model_decoder_hidden_size   = 1024
refined_model_decoder_num_layers    = 3
refined_model_encoder_embed_size    = 1024

# Refined model training settings
refined_model_batch_size            = 32
refined_model_num_workers           = 8
refined_model_num_epochs            = 2000
refined_model_save_every            = 1
refined_model_save_every_idx        = 2000
