# -*- coding: utf-8 -*-
import re
import numpy as np
import torch
import torch.nn as nn
# import nlp
import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd
import evaluate
import datasets
from datasets import Dataset, DatasetDict, Features

metric = evaluate.load("seqeval", cache_dir="./cache/")
model_checkpoint = "roberta-large"
max_lenght = 512

print(f"Combined - 20 epochs - {model_checkpoint}")

target_dataset = "AAEC"

print("TRAIN SPAN DETECTION")

# Preprocess data
file_path = "./dataset/"

df_span_train = pd.read_csv(file_path+"span_train.csv", header=0, index_col=0)
# Train set for fine tuning
df_span_train_ft = df_span_train[(df_span_train.text_id.str.contains(target_dataset)) | (df_span_train.text_id.str.contains("ArGPT"))]
df_span_train_ft["tokens"] = df_span_train_ft.tokens.apply(lambda x: x[1:-1].split(', '))
df_span_train_ft["tokens"] = df_span_train_ft["tokens"].apply(lambda x: [i.replace("'", "").strip() for i in x])
df_span_train_ft["chunk_tags"] = df_span_train_ft.chunk_tags.apply(lambda x: x[1:-1].split(','))
df_span_train_ft["chunk_tags"] = df_span_train_ft["chunk_tags"].apply(lambda x: [int(i, 32) for i in x])

# Train set for pretraining
df_span_train = df_span_train[(df_span_train.text_id.str.contains(target_dataset)) | (df_span_train.text_id.str.contains("ArGPT")) | (df_span_train.text_id.str.contains("AbstRCT")) | (df_span_train.text_id.str.contains("AASD")) | (df_span_train.text_id.str.contains("CDCP")) | (df_span_train.text_id.str.contains("MTC"))]
df_span_train["tokens"] = df_span_train.tokens.apply(lambda x: x[1:-1].split(', '))
df_span_train["tokens"] = df_span_train["tokens"].apply(lambda x: [i.replace("'", "").strip() for i in x])
df_span_train["chunk_tags"] = df_span_train.chunk_tags.apply(lambda x: x[1:-1].split(','))
df_span_train["chunk_tags"] = df_span_train["chunk_tags"].apply(lambda x: [int(i, 32) for i in x])

df_span_test = pd.read_csv(file_path+"span_test.csv", header=0, index_col=0)
# Test set containing only ArGPT
df_span_test_argpt = df_span_test[df_span_test.text_id.str.contains("ArGPT")]
df_span_test_argpt["tokens"] = df_span_test_argpt.tokens.apply(lambda x: x[1:-1].split(', '))
df_span_test_argpt["tokens"] = df_span_test_argpt["tokens"].apply(lambda x: [i.replace("'", "").strip() for i in x])
df_span_test_argpt["chunk_tags"] = df_span_test_argpt.chunk_tags.apply(lambda x: x[1:-1].split(','))
df_span_test_argpt["chunk_tags"] = df_span_test_argpt["chunk_tags"].apply(lambda x: [int(i, 32) for i in x])
test_text_ids_argpt = df_span_test_argpt.text_id.to_list()

# Test set containing only AAEC
df_span_test = df_span_test[df_span_test.text_id.str.contains(target_dataset)]
df_span_test["tokens"] = df_span_test.tokens.apply(lambda x: x[1:-1].split(', '))
df_span_test["tokens"] = df_span_test["tokens"].apply(lambda x: [i.replace("'", "").strip() for i in x])
df_span_test["chunk_tags"] = df_span_test.chunk_tags.apply(lambda x: x[1:-1].split(','))
df_span_test["chunk_tags"] = df_span_test["chunk_tags"].apply(lambda x: [int(i, 32) for i in x])
test_text_ids = df_span_test.text_id.to_list()

df_span_val = pd.read_csv(file_path+"span_val.csv", header=0, index_col=0)
df_span_val = df_span_val[(df_span_val.text_id.str.contains(target_dataset))| (df_span_train.text_id.str.contains("ArGPT"))]
df_span_val["tokens"] = df_span_val.tokens.apply(lambda x: x[1:-1].split(', '))
df_span_val["tokens"] = df_span_val["tokens"].apply(lambda x: [i.replace("'", "").strip() for i in x])
df_span_val["chunk_tags"] = df_span_val.chunk_tags.apply(lambda x: x[1:-1].split(','))
df_span_val["chunk_tags"] = df_span_val["chunk_tags"].apply(lambda x: [int(i, 32) for i in x])

df_span_train = df_span_train.reset_index(drop=True)
df_span_train_ft = df_span_train_ft.reset_index(drop=True)
df_span_test = df_span_test.reset_index(drop=True)
df_span_test_argpt = df_span_test_argpt.reset_index(drop=True)
df_span_val = df_span_val.reset_index(drop=True)


# Create Dataset I for span detection

features = Features(
    (
      {
          "text_id": datasets.Value("string"),
          "tokens": datasets.Sequence(datasets.Value("string")),
          "chunk_tags":datasets.Sequence(
              datasets.features.ClassLabel(
                  names=[
                      "O",
                      "B-ARG",
                      "I-ARG",
                  ]
              )
          )
      }
        )
  )

train_ds = Dataset.from_pandas(df_span_train, features=features)
train_ds_ft= Dataset.from_pandas(df_span_train_ft, features=features)
test_ds = Dataset.from_pandas(df_span_test, features=features)
test_ds_argpt = Dataset.from_pandas(df_span_test_argpt, features=features)
val_ds =  Dataset.from_pandas(df_span_val, features=features)
original_data_span = DatasetDict()
original_data_span["train_ft"] = train_ds_ft
original_data_span["train"] = train_ds
original_data_span["test"] = test_ds
original_data_span["test_argpt"] = test_ds_argpt
original_data_span["val"] = val_ds

del df_span_train
del df_span_train_ft
del df_span_test_argpt
del df_span_test
del df_span_val

task_feature = original_data_span["train"].features["chunk_tags"]
label_names = task_feature.feature.names
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

from transformers import (
   AutoModelForTokenClassification, AutoTokenizer
   )

tokenizer = AutoTokenizer.from_pretrained(
    model_checkpoint,
    add_prefix_space=True,
)

import numpy as np

def extend(a):
    out = []
    for sublist in a:
        out.extend(sublist)
        out.append("O")
    return [out]


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=extend(true_predictions), references=extend(true_labels))
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], is_split_into_words=True, truncation=True, padding="max_length", max_length=max_lenght,
    )
    all_labels = examples["chunk_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

labels = original_data_span["train"][0]["chunk_tags"]

tokenized_training_input = original_data_span.map(
    tokenize_and_align_labels,
    batched=True,
    #num_proc=accelerator.num_processes,
    remove_columns=original_data_span["train"].column_names,
)

from transformers import TrainingArguments, Trainer

model =  AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)

filepath = "./models/Span"

args = TrainingArguments(
    output_dir=filepath,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=100,
    learning_rate=2e-6,
    weight_decay=0.01,
    seed=9,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_training_input["train"],
    eval_dataset=tokenized_training_input["val"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
print(trainer.train_dataset)
print(trainer.eval_dataset)
trainer.train()
# If we need fine tuning
#print("Fine Tuning")
#trainer.train_dataset=tokenized_training_input["train_ft"]
#trainer.eval_dataset=tokenized_training_input["val_2"]
#trainer.train()

span_results = trainer.evaluate(tokenized_training_input["test"])
span_results_argpt = trainer.evaluate(tokenized_training_input["test_argpt"])

print(f"SPAN RESULTS AAEC: {span_results}")
print(f"\n SPAN RESULTS ArGPT: {span_results_argpt}")

# To create a file containing all the components the Span Detection Model extracted from each text
print("Genereating components file")

def get_components(spans, text, attention, test_text_ids, tokenizer=tokenizer):
    # Function to construct the components extracted from the text based on the predicted BIO tags
    assert len(spans) == len(text)
    predicted_components = list()

    component = list()
    for j, span in enumerate(spans):
        assert len(span) == len(text[j])
        assert len(span) == len(attention[j])
        former_label = 0
        component = list()
        for i, label in enumerate(span):
            label = int(label)
            if int(attention[j][i]) == 0:
                continue
            elif former_label == 0 and label == 0:
                continue
            elif former_label == 0 and label == 1:
                decoded = tokenizer.decode(text[j][i]).strip().replace(" ", "")
                component.append(decoded)
            elif former_label in (1,2) and label == 2:
                decoded = tokenizer.decode(text[j][i]).strip().replace(" ", "")
                component.append(decoded)
            elif former_label in (1,2) and label == 0:
                if len(component) > 0:
                    predicted_components.append({
                        "text_id": test_text_ids[j],
                        "component_tokens": " ".join(component).replace(" ,", ",")
                    })
                    component = list()
            elif former_label in (1,2) and label == 1:
                if len(component) > 0:
                    predicted_components.append({
                        "text_id": test_text_ids[j],
                        "component_tokens": " ".join(component).replace(" ,", ",")
                    })
                    component = list()
            former_label = label
        if len(component) > 0:
            predicted_components.append({
                "text_id": test_text_ids[j],
                "component_tokens": " ".join(component).replace(" ,", ",")
            })
            component = list()
    return predicted_components


# Generates file for AAEC
spans = trainer.predict(tokenized_training_input["test"])
logits, _, _ = spans
predictions = np.argmax(logits, axis=-1)
components = get_components(predictions, list(tokenized_training_input["test"]["input_ids"]), list(tokenized_training_input["test"]["attention_mask"]), test_text_ids)
df_predicted_components = pd.DataFrame.from_dict(components)
df_predicted_components["tokens"] = df_predicted_components.component_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())

# Generates file for ArGPT
spans = trainer.predict(tokenized_training_input["test_argpt"])
logits, _, _ = spans
predictions = np.argmax(logits, axis=-1)
components = get_components(predictions, list(tokenized_training_input["test_argpt"]["input_ids"]), list(tokenized_training_input["test_argpt"]["attention_mask"]), test_text_ids_argpt)
df_predicted_components_argpt = pd.DataFrame.from_dict(components)
df_predicted_components_argpt["tokens"] = df_predicted_components_argpt.component_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())

print("Genereating file")
path = "./results/Example/"
df_predicted_components = df_predicted_components[["text_id", "component_tokens"]]
print(df_predicted_components.shape)
df_predicted_components_argpt = df_predicted_components_argpt[["text_id", "component_tokens"]]
print(df_predicted_components_argpt.shape)
df_final = pd.concat([df_predicted_components, df_predicted_components_argpt], ignore_index=True)
print(df_final.shape)
df_final.to_csv(f"{path}/span_results.csv")
