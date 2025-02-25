import os
import torch
from tqdm import tqdm
from datasets import load_dataset
import sentencepiece as spm
from configs import ModelConfig
def train_tokenizer(dataset, vocab_size=32000, train_tokens=70_000_000):
    """Train the tokenizer on a subset of data"""
    print("Preparing data for tokenizer training...")
    total_tokens = 0
    with open('temp_train.txt', 'w', encoding='utf-8') as f:
        for example in tqdm(dataset):
            text = example['text']
            estimated_tokens = len(text.split())  # rough estimation
            if total_tokens + estimated_tokens > train_tokens:
                break
            f.write(text + '\n')
            total_tokens += estimated_tokens

    print(f"Training tokenizer on approximately {total_tokens:,} tokens...")
    spm.SentencePieceTrainer.train(
        input='temp_train.txt',
        model_prefix='tokenizer_model',
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=1.0,
        max_sentence_length=ModelConfig.block_size
    )

    # Load the trained tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load('tokenizer_model.model')
    
    # Clean up
    # if os.path.exists('temp_train.txt'):
    #     os.remove('temp_train.txt')
    
    return sp

def prepare_openwebtext(vocab_size=32000, token_limit=1_000_000_000, chunk_size=100000):
    """Main function to prepare the dataset"""
    # Load dataset in streaming mode
    dataset = load_dataset("openwebtext", streaming=True)
    train_stream = dataset['train'].shuffle(seed=2357)
    # 
    # First phase: Train tokenizer
    tokenizer = train_tokenizer(train_stream, vocab_size)
    
    # Reset stream for full data processing
    train_stream = dataset['train'].shuffle(seed=2357)
    
    # Second phase: Process full dataset
    print("\nProcessing full dataset...")
    train_chunks = []
    val_chunks = []
    total_tokens = 0
    val_size = 10000  # Number of validation examples
    is_val = True
    
    for example in tqdm(train_stream):
        ids = tokenizer.encode_as_ids(example['text'])
        
        if is_val and len(val_chunks) < val_size:
            val_chunks.extend(ids)
            if len(val_chunks) >= val_size:
                is_val = False
        else:
            if total_tokens + len(ids) > token_limit:
                break
            train_chunks.extend(ids)
            total_tokens += len(ids)
            
            # Status update and tensor conversion at checkpoints
            if len(train_chunks) >= chunk_size:
                print(f"\nProcessed {total_tokens:,} tokens...")

    # Convert final chunks to tensors
    print("\nConverting to tensors...")
    train_data = torch.tensor(train_chunks, dtype=torch.long)
    val_data = torch.tensor(val_chunks, dtype=torch.long)

    print(f"Train data size: {train_data.size()}")
    print(f"Validation data size: {val_data.size()}")
    print(f"Total tokens in training: {total_tokens:,}")
    print(f"Vocabulary size: {tokenizer.get_piece_size()}")

    return train_data, val_data, tokenizer
    # return tokenizer