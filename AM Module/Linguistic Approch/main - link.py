# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd
import evaluate
import datasets
from datasets import Dataset, DatasetDict, Features

metric = evaluate.load("seqeval", cache_dir="./cache/")
model_checkpoint = "roberta-large"
max_lenght = 512

print(f"20 epochs - {model_checkpoint}")

target_dataset = "AAEC"
file_path = "./dataset/"

print("TRAINING LINK Detection")

file_path = "./dataset/"
# LINK DETECTION
df_component_train = pd.read_csv(file_path+"component_train.csv", header=0, index_col=0)[["text_id", "text_tokens"]]
df_component_train = df_component_train[df_component_train.text_id.str.contains(target_dataset)]
df_component_train = df_component_train.drop_duplicates()

df_component_test = pd.read_csv(file_path+"component_test.csv", header=0, index_col=0)[["text_id", "text_tokens"]]
df_component_test = df_component_test[df_component_test.text_id.str.contains(target_dataset)]
df_component_test = df_component_test.drop_duplicates()

df_component_val = pd.read_csv(file_path+"component_val.csv", header=0, index_col=0)[["text_id", "text_tokens"]]
df_component_val = df_component_val[df_component_val.text_id.str.contains(target_dataset)]
df_component_val = df_component_val.drop_duplicates()

# Load Link Detection data
df_link_train = pd.read_csv(file_path+"link_train.csv", header=0, index_col=0)[["text_id", "source_tokens", "target_tokens", "labels"]]
df_link_train = df_link_train[df_link_train.text_id.str.contains(target_dataset)]
df_link_train["labels"] = df_link_train["labels"].replace("Link", 1)
df_link_train["labels"] = df_link_train["labels"].replace("None", 0)
df_link_train = pd.merge(df_link_train, df_component_train,  how='left', left_on=['text_id'], right_on = ['text_id'])

df_link_test = pd.read_csv(file_path+"link_test.csv", header=0, index_col=0)[["text_id", "source_tokens", "target_tokens", "labels"]]
df_link_test = df_link_test[df_link_test.text_id.str.contains(target_dataset)]
df_link_test["labels"] = df_link_test["labels"].replace("Link", 1)
df_link_test["labels"] = df_link_test["labels"].replace("None", 0)
df_link_test = pd.merge(df_link_test, df_component_test,  how='left', left_on=['text_id'], right_on = ['text_id'])

df_link_val = pd.read_csv(file_path+"link_val.csv", header=0, index_col=0)[["text_id", "source_tokens", "target_tokens", "labels"]]
df_link_val = df_link_val[df_link_val.text_id.str.contains(target_dataset)]
df_link_val["labels"] = df_link_val["labels"].replace("Link", 1)
df_link_val["labels"] = df_link_val["labels"].replace("None", 0)
df_link_val = pd.merge(df_link_val, df_component_val,  how='left', left_on=['text_id'], right_on = ['text_id'])

df_link_train = df_link_train.reset_index(drop=True)
df_link_test = df_link_test.reset_index(drop=True)
df_link_val = df_link_val.reset_index(drop=True)

print(df_link_test.head())

# Create Dataset for link detection
features = Features(
    (
      {
          "text_id": datasets.Value("string"),
          "source_tokens": datasets.Value("string"),
          "target_tokens": datasets.Value("string"),
          "text_tokens": datasets.Value("string"),
          "labels": datasets.features.ClassLabel(
                  names=[
                      "None",
                      "Link"
                  ]
              )
      }
        )
  )
train_ds = Dataset.from_pandas(df_link_train, features=features)
test_ds = Dataset.from_pandas(df_link_test, features=features)
val_ds = Dataset.from_pandas(df_link_val, features=features)
original_data_link = DatasetDict()
original_data_link["train"] = train_ds
original_data_link["test"] = test_ds
original_data_link["val"] = val_ds

num_none = df_link_train[df_link_train["labels"] == 0].shape[0]
num_link = df_link_train[df_link_train["labels"] == 1].shape[0]
train_link_weights=torch.tensor([1.0, float(num_none/num_link)])

task_feature = original_data_link["train"].features["labels"]
label_names = task_feature.names
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
     DataCollatorWithPadding, TextClassificationPipeline
    )

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

model_relation = AutoModelForSequenceClassification.from_pretrained(
  model_checkpoint, id2label=id2label, label2id=label2id,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def preprocess_function(examples):
    relations = list()
    for i, source in enumerate(examples['source_tokens']):
        left = f"{source} <\s> {examples['target_tokens'][i]}"
        relations.append(left)
    return tokenizer(relations, examples["text_tokens"], padding="max_length", max_length=max_lenght, truncation=True)

tokenized_training_input = original_data_link.map(preprocess_function, batched=True, remove_columns=["source_tokens", "target_tokens", "text_tokens"])

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import pandas as pd
from transformers import TrainingArguments, Trainer


def compute_metrics_relations(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(average="macro", y_true=labels, y_pred=pred)
    precision = precision_score(average="macro", y_true=labels, y_pred=pred)
    f1 = f1_score(average="micro", y_true=labels, y_pred=pred)
    f1 = f1_score(average="micro", y_true=labels, y_pred=pred)
    f1_link = f1_score(average="binary", pos_label=1, y_true=labels, y_pred=pred)
    f1_macro = f1_score(average="macro", y_true=labels, y_pred=pred)
    f1_weighted = f1_score(average="weighted", y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "f1_link": f1_link, "f1_macro": f1_macro, "f1_weighted": f1_weighted}

filepath = "./models/Link"

link_args = TrainingArguments(
    output_dir=filepath,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=20,
    learning_rate=2e-6,
    weight_decay=0.01,
    seed=27,
    load_best_model_at_end=True,
    metric_for_best_model="f1_link",
    greater_is_better=True,
)

from torch import nn
import torch
from transformers import Trainer


class CustomTrainer(Trainer):
    # Function to weight the labels
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=train_link_weights.to("cuda"))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

link_trainer = CustomTrainer(
    model=model_relation,
    args=link_args,
    train_dataset=tokenized_training_input["train"],
    eval_dataset=tokenized_training_input["val"],
    data_collator=data_collator,
    compute_metrics=compute_metrics_relations,
)

# Train pre-trained model
print(link_trainer.train_dataset)
print(link_trainer.eval_dataset)
link_trainer.train()

# # Make predictions
link_results = link_trainer.evaluate(tokenized_training_input["test"])

print(f"Link Results: {link_results}")


# Function that can be used to create a file with the predictions
model_component = model_relation.to('cpu')
pipe = TextClassificationPipeline(model=model_component, tokenizer=tokenizer)


def create_predicted_labels_df(pipe, components, name, path= "./results/Example/"):

    id_to_label = {
        "0": "None", "1": "Link",
    }
    tokenizer_kwargs = {"padding": "max_length", "max_length": max_lenght, "truncation": True}
    classified_components = list()
    for component in components:
        input = {"text": f"{component['source_tokens']} <\s> {component['target_tokens']}", "text_pair": component["text_tokens"]}
        prediction = pipe(input, return_all_scores=True, **tokenizer_kwargs)
        #print(prediction)
        best_score = 0
        component_dict = dict()
        for pred in prediction:
            prob = pred["score"]
            if prob > best_score:
                best_score = prob
                label = pred["label"]
            component_dict[pred["label"]] = prob

        component_dict["text_id"] = component["text_id"]
        component_dict["source_tokens"] = component["source_tokens"]
        component_dict["target_tokens"] = component["target_tokens"]
        component_dict["text_tokens"] = component["text_tokens"]
        component_dict["labels"] = id_to_label[str(component["labels"])]
        component_dict["predicted_labels"] = label
        classified_components.append(component_dict)

    path = "./results/Example/"
    df_classified_components = pd.DataFrame.from_dict(classified_components)
    df_classified_components.to_csv(f"{path}/{name}.csv")
    return df_classified_components
