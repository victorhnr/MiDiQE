# -*- coding: utf-8 -*-
import os
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./cache/"
os.environ["TRANSFORMERS_CACHE"] = "./cache/"
os.environ["HF_HOME"] = "./cache/"
import numpy as np
import torch
import torch.nn as nn
import logging
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
import statistics
logging.basicConfig(level=logging.INFO)

torch.cuda.empty_cache()
print(f"Is cuda available: {torch.cuda.is_available()}")
print(f"Number of available GPUs: {torch.cuda.device_count()}")

import pandas as pd
import datasets
import transformers
from datasets import Dataset, DatasetDict, Features
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer
)

NUM_EPOCHS = 20

# Max size was determined by what could fit in the memory
models = {
    "microsoft/deberta-v3-large": 512,
    "allenai/longformer-base-4096": 1024,
    "allenai/longformer-large-4096": 700,
    "roberta-base": 512,
    "roberta-large": 512,
}
seeds = [0, 9, 27, 42, 1871, 1994, 2000, 2005, 2008, 2023]
batch_size = 4

print("Training Argument Quality Task")

file_path = "./dataset/"
target_dataset = "AAEC"

# Argument Quality

file_path = file_path + "/argument_quality.csv"
df_argument_quality = pd.read_csv(file_path, header=0)
df_argument_quality = df_argument_quality.drop(columns=["Unnamed: 0"], axis=1)
df_argument_quality.rename(columns={"tokens": "text_tokens"}, inplace=True)
df_argument_quality.rename(columns={"arg_quality": "labels"}, inplace=True)
conditions = [
  (df_argument_quality["labels"] == "good"),
  (df_argument_quality["labels"] == "ugly"),
]
choices = ["good", "good"]
df_argument_quality["labels"] = np.select(conditions, choices, default="bad")
df_argument_quality["labels"] = df_argument_quality["labels"].replace("bad", 1)
df_argument_quality["labels"] = df_argument_quality["labels"].replace("good", 0)

df_argument_quality_train = df_argument_quality[df_argument_quality["split"] == "TRAIN"]
df_argument_quality_test = df_argument_quality[df_argument_quality["split"] == "TEST"]
df_argument_quality_val = df_argument_quality[df_argument_quality["split"] == "VAL"]
df_argument_quality_train = df_argument_quality_train.drop(columns=["split"], axis=1)
df_argument_quality_test = df_argument_quality_test.drop(columns=["split"], axis=1)
df_argument_quality_val = df_argument_quality_val.drop(columns=["split"], axis=1)
df_argument_quality_train = df_argument_quality_train[["text_id", "text_tokens", "labels"]]
df_argument_quality_test = df_argument_quality_test[["text_id", "text_tokens", "labels"]]
df_argument_quality_val = df_argument_quality_val[["text_id", "text_tokens", "labels"]]

df_quality_train = df_argument_quality_train.reset_index(drop=True)
df_quality_val = df_argument_quality_val.reset_index(drop=True)
df_quality_test = df_argument_quality_test.reset_index(drop=True)

# Create Dataset II for component classificantion
features = Features(
    (
      {
          "text_id": datasets.Value("string"),
          "text_tokens": datasets.Value("string"),
          "labels": datasets.features.ClassLabel(
                  names=[
                    "good",
                    "bad"
                  ]
              ),
      }
        )
  )
train_ds = Dataset.from_pandas(df_quality_train, features=features)
test_ds = Dataset.from_pandas(df_quality_val, features=features)
val_ds = Dataset.from_pandas(df_quality_test, features=features)

original_data_quality = DatasetDict()
original_data_quality["train"] = train_ds
original_data_quality["test"] = test_ds
original_data_quality["val"] = val_ds

num_good = df_quality_train[df_quality_train["labels"] == 0].shape[0]
num_bad = df_quality_train[df_quality_train["labels"] == 1].shape[0]
train_component_weights=torch.tensor([1.0, float(num_good/num_bad)])
print(train_component_weights)

del df_quality_train
del df_quality_test
del df_quality_val

task_feature = original_data_quality["train"].features["labels"]
label_names = task_feature.names
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}


all_results = dict()

for key in models.keys():
    model_statistics = list()
    for SEED in seeds:
        print(f"Training {key} with seed {SEED}")
        model_checkpoint = key
        max_lenght = models[key]

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True, use_fast = False)

        model_quality = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, id2label=id2label, label2id=label2id,
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        def preprocess_function(examples):
            return tokenizer(examples["text_tokens"], padding="max_length", max_length=max_lenght, truncation=True
                            )

        tokenized_training_input = original_data_quality.map(preprocess_function, batched=True, remove_columns=["text_id", "text_tokens"])


        def compute_metrics_quality(p):
            pred, labels = p
            pred = np.argmax(pred, axis=1)

            accuracy = accuracy_score(y_true=labels, y_pred=pred)
            recall = recall_score(average="macro", y_true=labels, y_pred=pred)
            precision = precision_score(average="macro", y_true=labels, y_pred=pred)
            f1_macro = f1_score(average="macro", y_true=labels, y_pred=pred )
            f1 = f1_score(average="micro", y_true=labels, y_pred=pred )
            f1_weighted = f1_score(average="weighted", y_true=labels, y_pred=pred)

            return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


        filepath = f"./models/Quality/{key}"

        quality_args = transformers.TrainingArguments(
            output_dir=filepath,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=20,
            learning_rate=2e-6,
            weight_decay=0.01,
            seed=SEED,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
        )

        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                # forward pass
                outputs = model(**inputs)
                logits = outputs.get("logits")
                # compute custom loss (suppose one has 3 labels with different weights)
                loss_fct = nn.CrossEntropyLoss(weight=train_component_weights.to("cuda"))
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        quality_trainer = CustomTrainer(
            model=model_quality,
            args=quality_args,
            train_dataset=tokenized_training_input["train"],
            eval_dataset=tokenized_training_input["val"],
            data_collator=data_collator,
            compute_metrics=compute_metrics_quality,
        )

        # Train pre-trained model
        print(quality_trainer.train_dataset)
        print(quality_trainer.eval_dataset)
        quality_trainer.train()

        # Make predictions
        quality_results = quality_trainer.evaluate(tokenized_training_input["test"])

        print(f"Quality {SEED} Results: {quality_results}")
        model_statistics.append(quality_results["eval_f1_macro"])
    
    mean = statistics.mean(model_statistics)
    deviation = statistics.stdev(model_statistics)
    all_results[key] = f"Macro mean: {mean}, Macro standard deviation: {deviation}"
    print(f"\n {key} Results: {all_results[key]}")


print("\n Final Results \n")

for key in all_results.keys():
    print(f"\n {key} Results: {all_results[key]}")
