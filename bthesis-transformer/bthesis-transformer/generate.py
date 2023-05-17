import random
import fire
import torch.nn as nn
import torch
import torch.optim.lr_scheduler as lr_scheduler
import tqdm
import transformers
import utils

NUM_TOKENS = 256

def main(n=1000, a=1e-3, k=256, b=128, h=2, d=2, s=256, f=False):
    num_batches, alpha, emb, batch_size, heads, depth, seq_len, final = int(n), float(a), int(k), int(b), int(h), int(d), int(s), bool(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the enwik8 dataset
    train_data, val_data, test_data = utils.enwik8("../data/enwik8.gz")
    train_data, test_data = (torch.cat([train_data, val_data], dim=0), test_data) if final else (train_data, val_data)

    # Create the model
    model = transformers.GrtTransformer(emb, heads, depth, seq_len, NUM_TOKENS)
    model.to(device)

    # Train and evaluate the model
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    loss_fn = nn.CrossEntropyLoss()

    instance_count = 0
    for i in tqdm.trange(num_batches):  # tqdm is a progress bar
        optimizer.zero_grad()

        x, y = utils.sample_batch(train_data, seq_len, batch_size)
        x, y = x.to(device), y.to(device)

        instance_count += x.size(0)

        y_hat = model(x)
        loss = loss_fn(y_hat, y)

        loss.backward()

        optimizer.step()

        scheduler.step()

        if i != 0 and (i % 10 == 0 or i == num_batches - 1):
            with torch.no_grad():
                # Sample and print a random sequence
                # Slice a random seed from the test data and sample some model output for it
                seedfr = random.randint(0, test_data.size(0) - seq_len)
                seed = test_data[seedfr:seedfr + seq_len].to(torch.long)

                seed = seed.to(device)

                utils.sample_sequence(model, seed=seed, max_context=seq_len, length=600, verbose=True)

                # TODO: bits per byte maybe?


if __name__ == '__main__':
    fire.Fire(main)