# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
# import nlp
import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd
import datasets
from datasets import Dataset, DatasetDict, Features

model_checkpoint = "roberta-large"
max_lenght = 512

print(f"20 epochs - {model_checkpoint}")

target_dataset = "AAEC"

print("Training Component Classification")

file_path = "./dataset/"

# Preprocess components

# COMPONENT CLASSIFICATION
df_component_train = pd.read_csv(file_path+"component_train.csv", header=0, index_col=0)[["text_id", "component_tokens", "text_tokens", "labels"]]
df_component_train = df_component_train[df_component_train.text_id.str.contains(target_dataset)]
df_component_train["labels"] = df_component_train["labels"].replace("MajorClaim", 2)
df_component_train["labels"] = df_component_train["labels"].replace("Claim", 1)
df_component_train["labels"] = df_component_train["labels"].replace("Premise", 0)

df_component_test = pd.read_csv(file_path+"component_test.csv", header=0, index_col=0)[["text_id", "component_tokens", "text_tokens", "labels"]]
df_component_test.rename(columns={"minimalist_labels": "labels"}, inplace=True)
df_component_test["labels"] = df_component_test["labels"].replace("MajorClaim", 2)
df_component_test["labels"] = df_component_test["labels"].replace("Claim", 1)
df_component_test["labels"] = df_component_test["labels"].replace("Premise", 0)

df_component_val = pd.read_csv(file_path+"component_val.csv", header=0, index_col=0)[["text_id", "component_tokens", "text_tokens", "labels"]]
df_component_val = df_component_val[df_component_val.text_id.str.contains(target_dataset)]
df_component_val["labels"] = df_component_val["labels"].replace("MajorClaim", 2)
df_component_val["labels"] = df_component_val["labels"].replace("Claim", 1)
df_component_val["labels"] = df_component_val["labels"].replace("Premise", 0)

df_component_train = df_component_train.reset_index(drop=True)
df_component_test = df_component_test.reset_index(drop=True)
df_component_val = df_component_val.reset_index(drop=True)

# Create Dataset II for component classificantion
features = Features(
    (
      {
          "text_id": datasets.Value("string"),
          "component_tokens": datasets.Value("string"),
          "text_tokens": datasets.Value("string"),
          "labels": datasets.features.ClassLabel(
                  names=[
                    "Premise",
                    "Claim",
                    "MajorClaim",
                  ]
              ),
      }
        )
  )
train_ds = Dataset.from_pandas(df_component_train, features=features)
test_ds = Dataset.from_pandas(df_component_test, features=features)
val_ds = Dataset.from_pandas(df_component_val, features=features)
original_data_component = DatasetDict()
original_data_component["train"] = train_ds
original_data_component["test"] = test_ds
original_data_component["val"] = val_ds

num_premise = df_component_train[df_component_train["labels"] == 0].shape[0]
num_c = df_component_train[df_component_train["labels"] == 1].shape[0]
num_mc = df_component_train[df_component_train["labels"] == 2].shape[0]

train_component_weights=torch.tensor([1.0, float(num_premise/num_c), float(num_premise/num_mc)])

task_feature = original_data_component["train"].features["labels"]
label_names = task_feature.names
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
     DataCollatorWithPadding, TextClassificationPipeline
    )

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

model_component = AutoModelForSequenceClassification.from_pretrained(
  model_checkpoint, id2label=id2label, label2id=label2id,
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def preprocess_function(examples):
    return tokenizer(examples["component_tokens"], examples["text_tokens"], padding="max_length", max_length=max_lenght, truncation="only_second"
                     )

tokenized_training_input = original_data_component.map(preprocess_function, batched=True, remove_columns=["component_tokens", "text_tokens"])

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import pandas as pd
from transformers import TrainingArguments, Trainer


def compute_metrics_component(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(average="macro", y_true=labels, y_pred=pred)
    precision = precision_score(average="macro", y_true=labels, y_pred=pred)
    f1_macro = f1_score(average="macro", y_true=labels, y_pred=pred )
    f1 = f1_score(average="micro", y_true=labels, y_pred=pred )
    f1_weighted = f1_score(average="weighted", y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


filepath = "./models/Component"

component_args = TrainingArguments(
    output_dir=filepath,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=20,
    learning_rate=2e-6,
    weight_decay=0.01,
    seed=9,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True
)

from torch import nn
import torch
from transformers import Trainer


class CustomTrainer(Trainer):
     def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
         # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
#         # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=train_component_weights.to("cuda"))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


component_trainer = CustomTrainer(
     model=model_component,
     args=component_args,
     train_dataset=tokenized_training_input["train"],
     eval_dataset=tokenized_training_input["val"],
     data_collator=data_collator,
     compute_metrics=compute_metrics_component,
)

# Train pre-trained model
component_trainer.train()

## Make predictions
component_results = component_trainer.evaluate(tokenized_training_input["test"])

print(f"Component Results: {component_results}")

model_component = model_component.to('cpu')
pipe = TextClassificationPipeline(model=model_component, tokenizer=tokenizer)


def create_predicted_labels_df(pipe, components, name, path= "./results/Example/"):
    # Function that can be used to create a files with all the predictions
    id_to_label = {
        "0": "Premise", "1": "Claim", "2": "MajorClaim"
    }
    tokenizer_kwargs = {"padding": "max_length", "max_length": max_lenght, "truncation": "only_second"}
    classified_components = list()
    for component in components:
        input = {"text": component["component_tokens"], "text_pair": component["text_tokens"]}
        prediction = pipe(input, return_all_scores=True, **tokenizer_kwargs)
        best_score = 0
        component_dict = dict()
        for pred in prediction:
            prob = pred["score"]
            if prob > best_score:
                best_score = prob
                label = pred["label"]
            component_dict[pred["label"]] = prob

        component_dict["text_id"] = component["text_id"]
        component_dict["component_tokens"] = component["component_tokens"]
        component_dict["labels"] = id_to_label[str(component["labels"])]
        component_dict["predicted_labels"] = label
        classified_components.append(component_dict)

    path = "./results/Example/"
    df_classified_components = pd.DataFrame.from_dict(classified_components)
    df_classified_components.to_csv(f"{path}/{name}.csv")
    return df_classified_components

