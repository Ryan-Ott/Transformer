import os

import fire
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from bttransformer import transformers, utils
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


VOCAB_SIZE = 2**15  # 32768
CLIP_VALUE = 1.0


def main(epochs=3, bsize=32, emb=128, eheads=4, ehidden=4, edrop=0.1, edepth=2, dheads=4, dhidden=2, ddrop=0.1, ddepth=4, set="train", final=False, v=False):
    """Train a transformer model on the XSum dataset."""
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the XSum dataset from HuggingFace
    dataset = load_dataset("xsum")

    # Train, validation and test splits
    train_x, train_y = dataset["train"]["document"], dataset["train"]["summary"]
    val_x, val_y = dataset["validation"]["document"], dataset["validation"]["summary"]
    test_x, test_y = dataset["test"]["document"], dataset["test"]["summary"]

    # Create a new tokenizer if missing, else load the existing one
    if os.path.exists("./data/xsum_tokenizer.json"):
        print("Using existing tokenizer")
        tokenizer = Tokenizer.from_file("./data/xsum_tokenizer.json")
    else:
        print("Creating new tokenizer")
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordPieceTrainer(vocab_size=VOCAB_SIZE , special_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]"])  # MASK token is not needed I believe

        tokenizer.train_from_iterator(list(zip(train_x, train_y)), trainer=trainer)

        tokenizer.save("./data/xsum_tokenizer.json")
    
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    # Sort data by length of document
    train_x, train_y = utils.sort(train_x, train_y)
    val_x, val_y = utils.sort(val_x, val_y)
    test_x, test_y = utils.sort(test_x, test_y)

    # Remove documents where the document is less than 3 times the length of the summary
    train_x, train_y = utils.filter(train_x, train_y)

    # For checking purposes
    # for i in range(5):
    #     print(f"Document {i+1}: {train_x[i]}")
    #     print(f"Summary {i+1}: {train_y[i]}")
    #     print(f"Length of document {i+1}: {len(train_x[i])}")
    #     print(f"Length of summary {i+1}: {len(train_y[i])}")
    #     print()
    
    train_batches = utils.preprocess(train_x, train_y, tokenizer, bsize, device)
    val_batches = utils.preprocess(val_x, val_y, tokenizer, bsize, device)
    test_batches = utils.preprocess(test_x, test_y, tokenizer, bsize, device)

    if v:
        print(len(train_batches))
        print(f"Example decoded tokenised instance:")
        print("Document: ", tokenizer.decode(train_batches[420][0][0].tolist()))
        print("Summary: ", tokenizer.decode(train_batches[420][1][0].tolist()))

    # Maximum sequence length
    max_seq_len = max(max(len(x) for x in document) for document in [train_x, val_x, test_x])
    print(f"\nMaximum sequence length: {max_seq_len}")

    # Model instantiation
    model = transformers.SumTransformer(  # shallow but wide encoder, deep but narrow decoder
        emb_dim=emb,
        vocab_size=tokenizer.get_vocab_size(),
        max_len=max_seq_len,
        enc_heads=eheads, enc_hidden=ehidden, enc_dropout=edrop, enc_depth=edepth,
        dec_heads=dheads, dec_hidden=dhidden, dec_dropout=ddrop, dec_depth=ddepth).to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10)

    # Early stopping
    min_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    # Load checkpoint if it exists
    checkpoint_path = './checkpoints/checkpoint_latest.pth'
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # continue from the next epoch
        print(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        start_epoch = 0  # start from scratch

    # Directory to save model checkpoints
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training epochs
    num_epochs = epochs
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        total_train_loss = 0
        for batch in train_batches:
            optimizer.zero_grad()
            inputs, labels = batch
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_train_loss += loss.item()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=CLIP_VALUE)
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_batches)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_batches:
                inputs, labels = batch
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_batches)

        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, os.path.join(checkpoint_dir, f'checkpoint_{epoch+1}.pth'))

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break
    
    # Testing phase
    if final:
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch in test_batches:
                inputs, labels = batch
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(test_batches)
        print(f"Test Loss: {avg_test_loss:.4f}")


    # TODO: add warmup
    # TODO: create graphs for L2 gradient norm, weights and biases, loss curves

if __name__ == '__main__':
    fire.Fire(main)
    # for debugging
    # main(bsize=32, emb=4, eheads=2, ehidden=2, edrop=0.1, edepth=1, dheads=2, dhidden=2, ddrop=0.1, ddepth=1, set="train", v=True)
