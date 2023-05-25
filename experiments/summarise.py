import math
import os
import random
import sys
import fire
import torch

from bttransformer import transformers

from datasets import concatenate_datasets, load_dataset
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


MIN_LENGTH = 15  # Minimum number of tokens in a sequence
CLIP = 1  # Gradient clipping


class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class BucketSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, sort_lens, batch_size, shuffle=True, drop_last=False):
        super().__init__(data_source)
        self.sort_lens = sort_lens
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        indices = list(range(len(self.sort_lens)))
        batches = [indices[i: i + self.batch_size] for i in range(0, len(indices), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        if self.drop_last:
            return len(self.sort_lens) // self.batch_size
        else:
            return (len(self.sort_lens) + self.batch_size - 1) // self.batch_size


class EarlyStopping:
    def __init__(self, patience=3, path='./checkpoints/checkpoint.pt', delta=0):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



def split_data(dataset, train_split, val_split):
    # Shuffle the dataset
    dataset = dataset.shuffle()

    # Split the dataset
    train_size = int(len(dataset) * train_split)
    val_size = int(len(dataset) * val_split)
    test_size = len(dataset) - train_size - val_size

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size+val_size]
    test_dataset = dataset[train_size+val_size:]

    return train_dataset, val_dataset, test_dataset


def train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, epochs, clip, early_stopping):
    # Ensure 'graphs' directory exists
    if not os.path.exists('graphs'):
        os.makedirs('graphs')

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0

        # For storing norms
        l2_norms = []

        for batch_idx, (docs, sums) in enumerate(train_loader):
            docs = docs.to(device)
            sums = sums.to(device)
            
            # Forward pass
            output = model(docs, sums)

            # Clear the gradients
            optimizer.zero_grad()

            # Calculate the loss
            loss = loss_fn(output.view(-1, output.size(-1)), sums.view(-1))
            running_loss += loss.item()

            # Backward pass
            loss.backward()

            # Compute L2 norm of gradients and store
            l2_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    l2_norm += param_norm.item() ** 2
            l2_norms.append(l2_norm ** 0.5)

            # Gradient clipping
            clip_grad_norm_(model.parameters(), clip)

            # Update weights
            optimizer.step()

            # Update learning rate
            scheduler.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} batch {batch_idx} loss {loss.item()}")

        running_loss /= len(train_loader)
        train_losses.append(running_loss)
        print(f"Epoch {epoch} Training Loss: {running_loss}")

        # After each epoch, validate the model
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for docs, sums in val_loader:
                docs = docs.to(device)
                sums = sums.to(device)
                
                output = model(docs)
                loss = loss_fn(output.view(-1, output.size(-1)), sums.view(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"Validation loss after epoch {epoch}: {val_loss}")

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Plot L2 Norm
        plt.figure()
        plt.plot(l2_norms)
        plt.title('Gradient L2 Norm')
        plt.savefig(f'graphs/l2_norm_epoch_{epoch}.png')
        plt.close()

    print("Training completed.")

    # Plot Losses
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss curves')
    plt.legend()
    plt.savefig('graphs/loss_curves.png')
    plt.close()

    # Load the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    # Plot weights and biases
    for name, param in model.named_parameters():
        if param.requires_grad:
            plt.figure()
            plt.hist(param.data.cpu().numpy().flatten(), bins=100)
            plt.title(name)
            plt.savefig(f'graphs/{name.replace(".", "_")}.png')
            plt.close()



def test(model, test_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for docs, sums in test_loader:
            docs = docs.to(device)
            sums = sums.to(device)

            output = model(docs)

            loss = loss_fn(output.view(-1, output.size(-1)), sums.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)

    print(f"\nTest Loss: {avg_loss}")


def main(
    epochs=3, alpha=1e-4, bsize=32, emb=128, vocab_size=2**14, sched="onecycle", final=False, v=False,
    eheads=4, ehidden=4, edrop=0.1, edepth=2,
    dheads=4, dhidden=2, ddrop=0.1, ddepth=4):
    torch.set_printoptions(threshold=sys.maxsize)
    
    # Load the best available device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the reddit_tifu dataset from HuggingFace
    dataset_short = load_dataset("reddit_tifu", "short")
    dataset_long = load_dataset("reddit_tifu", "long")

    # Concatenate the 'long' and 'short' datasets
    dataset = concatenate_datasets([dataset_short["train"], dataset_long["train"]])
    
    # Split the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = split_data(dataset, train_split=0.8, val_split=0.1)

    # Create a new WordPiece tokenizer from the training data based on Huggingface's implementation if missing
    if os.path.exists("./data/tifu_tokenizer.json"):
        print("Using existing tokenizer...")
        tokenizer = Tokenizer.from_file("./data/tifu_tokenizer.json")
    else:
        print("Creating new tokenizer")
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordPieceTrainer(vocab_size=vocab_size , special_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]"])  # ! Make sure that UNK gets used down the line

        # ? Should I also include val and test data?
        texts = list(train_dataset['documents']) + list(train_dataset['tldr']) + list(train_dataset['title'])  # also include title
        tokenizer.train_from_iterator(texts, trainer=trainer)

        tokenizer.save("./data/tifu_tokenizer.json")
    
    # TODO: Add post-processor

    # Keep only the 'documents' and 'tldr' columns
    print("Removing unnecessary columns...")
    train_data = zip(train_dataset["documents"], train_dataset["tldr"])
    val_data = zip(val_dataset["documents"], val_dataset["tldr"])
    test_data = zip(test_dataset["documents"], test_dataset["tldr"])

    # Filter out instances where the document or tldr is less than min length
    print("Filtering out short documents and summaries...")
    train_data = [(doc, sum) for doc, sum in train_data if len(doc) >= MIN_LENGTH and len(sum) >= MIN_LENGTH]
    val_data = [(doc, sum) for doc, sum in val_data if len(doc) >= MIN_LENGTH and len(sum) >= MIN_LENGTH]
    test_data = [(doc, sum) for doc, sum in test_data if len(doc) >= MIN_LENGTH and len(sum) >= MIN_LENGTH]

    # Print the first doc and summary from the training data
    if v:
        print(f"Document: {train_data[0][0]}")
        print(f"Summary: {train_data[0][1]}")
        print("--------------------------------------------------")

    # Tokenize the data
    print("Tokenizing the data...")
    train_data = [(tokenizer.encode(doc).ids, tokenizer.encode(sum).ids) for doc, sum in train_data]
    val_data = [(tokenizer.encode(doc).ids, tokenizer.encode(sum).ids) for doc, sum in val_data]
    test_data = [(tokenizer.encode(doc).ids, tokenizer.encode(sum).ids) for doc, sum in test_data]

    # Create a PyTorch DataLoader
    print("Creating data loaders...")

    # Calculate the lengths of the documents
    train_lens = [len(d) for d, _ in train_data]
    val_lens = [len(d) for d, _ in val_data]
    test_lens = [len(d) for d, _ in test_data]

    # Create datasets
    train_dataset = TextDataset(train_data)
    val_dataset = TextDataset(val_data)
    test_dataset = TextDataset(test_data)

    # Create bucket samplers
    train_sampler = BucketSampler(train_dataset, train_lens, bsize)
    val_sampler = BucketSampler(val_dataset, val_lens, bsize)
    test_sampler = BucketSampler(test_dataset, test_lens, bsize)
    
    def collate_fn(batch):
        # Unzipping the batch
        docs, sums = zip(*batch)

        # Convert lists to tensors
        docs = [torch.LongTensor(doc) for doc in docs]
        sums = [torch.LongTensor(sum) for sum in sums]
        
        # Pad sequences
        docs = pad_sequence(docs, batch_first=True, padding_value=tokenizer.token_to_id("[PAD]"))
        sums = pad_sequence(sums, batch_first=True, padding_value=tokenizer.token_to_id("[PAD]"))
        
        return docs, sums

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=collate_fn)

    # Get the largest document and summary lengths
    max_doc_len = max(train_lens)

    # Print the last instance of the first batch of the training data
    if v:
        batch = next(iter(train_loader))
        docs, sums = batch
        print(f"Document: {docs[-1]}")
        print(f"Summary: {sums[-1]}")
    
    # Create the model
    print("\nCreating the model...")
    model = transformers.SumTransformer(emb, tokenizer.get_vocab_size(), max_doc_len, eheads, ehidden, edrop, edepth, dheads, dhidden, ddrop, ddepth).to(device)

    # Define the loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

    # Create the scheduler
    lr_schedulers = {
        "linear": lr_scheduler.LinearLR(optimizer, start_factor=0.0001, end_factor=0.001, total_iters=len(train_loader)*epochs),
        "constant": lr_scheduler.ConstantLR(optimizer),
        "onecycle": lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=len(train_loader)*epochs, pct_start=0.3, anneal_strategy="linear"),
        "invsqrt": lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/math.sqrt(epoch) if epoch > 0 else 1),
        "cosinedecay": lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs)
    }

    scheduler = lr_schedulers[sched]

    early_stopping = EarlyStopping(patience=3, path='checkpoint.pt')

    train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, epochs, CLIP, early_stopping)

    if final:
        test(model, test_loader, loss_fn, device)


if __name__ == '__main__':
    fire.Fire(main)