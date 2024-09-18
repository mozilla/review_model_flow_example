# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os

from metaflow import (
    FlowSpec,
    IncludeFile,
    Parameter,
    card,
    current,
    step,
    environment,
    kubernetes,
    pypi
)
from metaflow.cards import Markdown

class ReviewSentimentFlow(FlowSpec):
    """
    This flow is a template for you to use
    for orchestration of your model.
    """

    # This is an example of a parameter. You can toggle this when you call the flow
    # with python template_flow.py run --offline False
    offline_wandb = Parameter(
        "offline",
        help="Do not connect to W&B servers when training",
        type=bool,
        default=True,
    )

    @card(type="default")
    @step
    def start(self):
        """
        Each flow has a 'start' step.

        You can use it for collecting/preprocessing data or other setup tasks.
        """

        self.next(self.train)

    @card
    @environment(
        vars={
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
            "WANDB_ENTITY": os.getenv("WANDB_ENTITY"),
            "WANDB_PROJECT": os.getenv("WANDB_PROJECT"),
        }
    )
    @pypi(python="3.10.8",
          packages={
            "scikit-learn": "1.5.0",
            "wandb": "0.18.1",
            "datasets": "3.0.0",
            "numpy": "2.1.1",
            "tqdm": "4.66.5",
            "transformers": "4.44.2",
            })
    @kubernetes(image="registry.hub.docker.com/dexterp37/mlops-copilot-demo:v3", gpu=1)
    @step
    def train(self):
        """
        In this step you can train your model,
        save checkpoints and artifacts,
        and deliver data to Weights and Biases
        for experiment evaluation
        """
        import json
        import wandb
        import collections

        import datasets
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import tqdm
        import transformers

        config_as_dict = json.loads(self.example_config)
        print(f"The config file says: {config_as_dict.get('example_key')}")

        if not self.offline_wandb:
            tracking_run = wandb.init(project=os.getenv("WANDB_PROJECT"))
            wandb_url = tracking_run.get_url()
            current.card.append(Markdown("# Weights & Biases"))
            current.card.append(
                Markdown(f"Your training run is tracked [here]({wandb_url}).")
            )

        print("All set. Running training.")

        seed = 1234

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])
        transformer_name = "bert-base-uncased"

        tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_name)

        def tokenize_and_numericalize_example(example, tokenizer):
            ids = tokenizer(example["text"], truncation=True)["input_ids"]
            return {"ids": ids}

        train_data = train_data.map(
            tokenize_and_numericalize_example, fn_kwargs={"tokenizer": tokenizer}
        )
        test_data = test_data.map(
            tokenize_and_numericalize_example, fn_kwargs={"tokenizer": tokenizer}
        )
        test_size = 0.25

        train_valid_data = train_data.train_test_split(test_size=test_size)
        train_data = train_valid_data["train"]
        valid_data = train_valid_data["test"]

        train_data = train_data.with_format(type="torch", columns=["ids", "label"])
        valid_data = valid_data.with_format(type="torch", columns=["ids", "label"])
        test_data = test_data.with_format(type="torch", columns=["ids", "label"])

        def get_collate_fn(pad_index):
            def collate_fn(batch):
                batch_ids = [i["ids"] for i in batch]
                batch_ids = nn.utils.rnn.pad_sequence(
                    batch_ids, padding_value=pad_index, batch_first=True
                )
                batch_label = [i["label"] for i in batch]
                batch_label = torch.stack(batch_label)
                batch = {"ids": batch_ids, "label": batch_label}
                return batch

            return collate_fn

        def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
            collate_fn = get_collate_fn(pad_index)
            data_loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                shuffle=shuffle,
            )
            return data_loader

        batch_size = 8

        train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
        valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
        test_data_loader = get_data_loader(test_data, batch_size, pad_index)

        class Transformer(nn.Module):
            def __init__(self, transformer, output_dim, freeze):
                super().__init__()
                self.transformer = transformer
                hidden_dim = transformer.config.hidden_size
                self.fc = nn.Linear(hidden_dim, output_dim)
                if freeze:
                    for param in self.transformer.parameters():
                        param.requires_grad = False

            def forward(self, ids):
                # ids = [batch size, seq len]
                output = self.transformer(ids, output_attentions=True)
                hidden = output.last_hidden_state
                # hidden = [batch size, seq len, hidden dim]
                attention = output.attentions[-1]
                # attention = [batch size, n heads, seq len, seq len]
                cls_hidden = hidden[:, 0, :]
                prediction = self.fc(torch.tanh(cls_hidden))
                # prediction = [batch size, output dim]

        transformer = transformers.AutoModel.from_pretrained(transformer_name)
        output_dim = len(train_data["label"].unique())
        freeze = False

        model = Transformer(transformer, output_dim, freeze)
        lr = 1e-5

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)
        criterion = criterion.to(device)

        def train(data_loader, model, criterion, optimizer, device):
            model.train()
            epoch_losses = []
            epoch_accs = []
            for batch in tqdm.tqdm(data_loader, desc="training..."):
                ids = batch["ids"].to(device)
                label = batch["label"].to(device)
                prediction = model(ids)
                loss = criterion(prediction, label)
                accuracy = get_accuracy(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                epoch_accs.append(accuracy.item())
            return np.mean(epoch_losses), np.mean(epoch_accs)

        def evaluate(data_loader, model, criterion, device):
            model.eval()
            epoch_losses = []
            epoch_accs = []
            with torch.no_grad():
                for batch in tqdm.tqdm(data_loader, desc="evaluating..."):
                    ids = batch["ids"].to(device)
                    label = batch["label"].to(device)
                    prediction = model(ids)
                    loss = criterion(prediction, label)
                    accuracy = get_accuracy(prediction, label)
                    epoch_losses.append(loss.item())
                    epoch_accs.append(accuracy.item())
            return np.mean(epoch_losses), np.mean(epoch_accs)

        def get_accuracy(prediction, label):
            batch_size, _ = prediction.shape
            predicted_classes = prediction.argmax(dim=-1)
            correct_predictions = predicted_classes.eq(label).sum()
            accuracy = correct_predictions / batch_size
            return accuracy

        n_epochs = 3
        best_valid_loss = float("inf")

        metrics = collections.defaultdict(list)

        for epoch in range(n_epochs):
            train_loss, train_acc = train(
                train_data_loader, model, criterion, optimizer, device
            )
            valid_loss, valid_acc = evaluate(valid_data_loader, model, criterion, device)
            metrics["train_losses"].append(train_loss)
            metrics["train_accs"].append(train_acc)
            metrics["valid_losses"].append(valid_loss)
            metrics["valid_accs"].append(valid_acc)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), "transformer.pt")
            print(f"epoch: {epoch}")
            print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
            print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")

        model.load_state_dict(torch.load("transformer.pt"))

        test_loss, test_acc = evaluate(test_data_loader, model, criterion, device)
        print(f"test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}")

        def predict_sentiment(text, model, tokenizer, device):
            ids = tokenizer(text)["input_ids"]
            tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
            prediction = model(tensor).squeeze(dim=0)
            probability = torch.softmax(prediction, dim=-1)
            predicted_class = prediction.argmax(dim=-1).item()
            predicted_probability = probability[predicted_class].item()
            return predicted_class, predicted_probability

        text = "This film is terrible!"
        print(predict_sentiment(text, model, tokenizer, device))

        text = "This film is not terrible, it's great!"
        print(predict_sentiment(text, model, tokenizer, device))

        self.next(self.end)

    @step
    def end(self):
        """
        This is the mandatory 'end' step: it prints some helpful information
        to access the model and the used dataset.
        """
        print(
            f"""
            Flow complete.

            """
        )


if __name__ == "__main__":
    ReviewSentimentFlow()
