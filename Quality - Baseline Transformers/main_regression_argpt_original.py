# -*- coding: utf-8 -*-
import os
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./cache/"
os.environ["TRANSFORMERS_CACHE"] = "./cache/"
os.environ["HF_HOME"] = "./cache/"
import torch
import logging
from sklearn.metrics import mean_squared_error, cohen_kappa_score
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

NUM_EPOCHS = 80

# Max size was determined based on what could fit the memory
models = {
    "microsoft/deberta-v3-large": 512,
    "allenai/longformer-base-4096": 1024,
    "allenai/longformer-large-4096": 700,
    "roberta-base": 512,
    "roberta-large": 512,
}
seeds = [0, 9, 27, 42, 1871, 1994, 2000, 2005, 2023, 3100]
batch_size = 4

print("Training Argument Quality Task")

file_path = "./dataset/"
target_dataset = "ArGPT"

# Argument Quality
df_evaluation = pd.read_csv(f"{file_path}/text_evaluations.csv", header=0)
df_evaluation = df_evaluation[["text_id", "tokens", "criteria_7", "criteria_8", "criteria_9", "split"]]
df_evaluation["criteria_7"] = df_evaluation["criteria_7"].apply(lambda x: float(10*x))
df_evaluation["criteria_8"] = df_evaluation["criteria_8"].apply(lambda x: float(10*x))
df_evaluation["criteria_9"] = df_evaluation["criteria_9"].apply(lambda x: float(10*x))
df_evaluation["final_grade"] = df_evaluation["criteria_7"] + df_evaluation["criteria_8"] + df_evaluation["criteria_9"]
df_evaluation["final_grade"] = df_evaluation["final_grade"].apply(lambda x: float(x/75))
df_evaluation_train = df_evaluation[df_evaluation["split"] == "TRAIN"]
df_evaluation_test = df_evaluation[df_evaluation["split"] == "TEST"]
df_evaluation_val = df_evaluation[df_evaluation["split"] == "VAL"]
df_evaluation_train = df_evaluation_train.drop(columns=["split"], axis=1)
df_evaluation_test = df_evaluation_test.drop(columns=["split"], axis=1)
df_evaluation_val = df_evaluation_val.drop(columns=["split"], axis=1)
df_evaluation_train = df_evaluation_train.reset_index(drop=True)
df_evaluation_test = df_evaluation_test.reset_index(drop=True)
df_evaluation_val = df_evaluation_val.reset_index(drop=True)

df_quality_train = df_evaluation_train.reset_index(drop=True)
df_quality_val = df_evaluation_val.reset_index(drop=True)
df_quality_test = df_evaluation_test.reset_index(drop=True)

# Create Dataset II for component classificantion
features = Features(
    (
      {
          "text_id": datasets.Value("string"),
          "tokens": datasets.Value("string"),
          "final_grade": datasets.Value("float32")
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

del df_quality_train
del df_quality_test
del df_quality_val

all_results = dict()

for key in models.keys():
    model_statistics_qwk = list()
    model_statistics_mse = list()
    for SEED in seeds:
        print(f"Training {key} with seed {SEED}")
        model_checkpoint = key
        max_lenght = models[key]

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        model_quality = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=1
        )

        def preprocess_function(examples):
            label = examples["final_grade"]
            examples = tokenizer(examples["tokens"], truncation=True, padding="max_length", max_length=models[key])
            # Change this to real number
            examples["label"] = float(label)
            return examples

        tokenized_training_input = original_data_quality.map(preprocess_function, remove_columns=["text_id", "tokens", "final_grade"])

        def compute_metrics_for_regression(eval_pred):
            logits, labels = eval_pred
            labels = labels.reshape(-1, 1)
            qwk_logits = 1000*logits
            qwk_labels = 1000*labels
            qwk_logits = qwk_logits.astype(int)
            qwk_labels = qwk_labels.astype(int)

            mse = mean_squared_error(100*labels, 100*logits)
            qwk = cohen_kappa_score(qwk_labels, qwk_logits, weights="quadratic")

            return {"mse": mse, "qwk": qwk}


        filepath = f"./models/Quality/{key}"

        quality_args = transformers.TrainingArguments(
            output_dir=filepath,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=2e-5,
            weight_decay=0.01,
            seed=SEED,
            load_best_model_at_end=True,
            metric_for_best_model="qwk",
            greater_is_better=True,
        )

        quality_trainer = Trainer(
            model=model_quality,
            args=quality_args,
            train_dataset=tokenized_training_input["train"],
            eval_dataset=tokenized_training_input["val"],
            compute_metrics=compute_metrics_for_regression,
        )

        # Train pre-trained model
        print(quality_trainer.train_dataset)
        print(quality_trainer.eval_dataset)
        quality_trainer.train()

        # Make predictions
        quality_results = quality_trainer.evaluate(tokenized_training_input["test"])

        print(f"Quality {SEED} Results: {quality_results}")
        model_statistics_qwk.append(quality_results["eval_qwk"])
        model_statistics_mse.append(quality_results["eval_mse"])
    
    mean_qwk = statistics.mean(model_statistics_qwk)
    deviation_qwk = statistics.stdev(model_statistics_qwk)
    mean_mse = statistics.mean(model_statistics_mse)
    deviation_mse = statistics.stdev(model_statistics_mse)
    all_results[key] = f"QWK mean: {mean_qwk}, QWK standard deviation: {deviation_qwk}, MSE mean {mean_mse}, MSE deviation: {deviation_mse}"
    print(f"\n {key} Results: {all_results[key]}")


print("\n Final Results \n")

for key in all_results.keys():
    print(f"\n {key} Results: {all_results[key]}")
