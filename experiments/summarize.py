import os

import fire
import torch

from bttransformer import transformers, utils
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


VOCAB_SIZE = 2**15  # 32768


def main(b=32, k=128, h=4, set="train", v=False):
    """Train a summary generation model and evaluate the accuracy.
    
    Args:
        b (int, optional): Batch size. Defaults to 32.
        k (int, optional): Dimensions of the embedding vector. Defaults to 128.
        h (int, optional): Number of heads in the multi-head attention layer. Defaults to 4.
        set (str, optional): Which dataset to use between "train", "val" and "test". Default to "train".
        v (bool, optional): Verbose. Prints batch information and produces graphs. Defaults to False.
    """
    batch_size, emb_dim, heads = int(b), int(k), int(h)
    
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the XSum dataset from HuggingFace
    dataset = load_dataset("xsum")
    
    # TODO: strip non-ASCII characters, lowercase like in cramming paper?

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

        # Combine the documents and summaries into one list
        tokenizer.train_from_iterator(list(zip(train_x, train_y)), trainer=trainer)  # TODO: also add val and test? Documentation does but wouldn't test model's ability to generalize

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
    # val_x, val_y = utils.sort(val_x, val_y)
    # test_x, test_y = utils.sort(test_x, test_y)

    # TODO: remove instances where the document is empty?

    if set == "train":
        x, y = train_x, train_y
    elif set == "val":
        x, y = val_x, val_y
    elif set == "test":
        x, y = test_x, test_y
    else:
        raise ValueError("Invalid set name")
    
    batches = utils.preprocess(x, y, tokenizer, batch_size, device)

    if v:
        print(len(batches))
        print(f"Example decoded tokenised instance:")
        print("Document: ", tokenizer.decode(batches[420][0][0].tolist()))
        print("Summary: ", tokenizer.decode(batches[420][1][0].tolist()))

    # Maximum sequence length
    max_seq_len = max(max(len(x) for x in document) for document in [train_x, val_x, test_x])
    print(f"Maximum sequence length: {max_seq_len}")

    model = transformers.SumTransformer(  # shallow but wide encoder, deep but narrow decoder
        emb_dim=emb_dim,
        heads=heads,
        vocab_size=tokenizer.get_vocab_size(),
        seq_len=max_seq_len,
        enc_hidden=4, enc_dropout=0.1, enc_depth=1,
        dec_hidden=2, dec_dropout=0.1, dec_depth=4).to(device)
    
    # TODO: add learning rate scheduler
    # TODO: add warmup
    # TODO: add gradient clipping
    # TODO: add early stopping
    # TODO: create graphs for L2 gradient norm, weights and biases, loss curves

if __name__ == '__main__':
    fire.Fire(main)
