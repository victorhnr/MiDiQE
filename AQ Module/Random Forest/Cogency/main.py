import os
import re
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
from sklearn.ensemble import RandomForestClassifier


path = "./dataset"
target_dataset = "AAEC"
num_labels = 5 # Can be 5, 3 and 2

# Training
df_component_train = pd.read_csv(path+"component_rel_quality_train.csv", header=0)[["text_id", "component_tokens", "labels", "supports_strenght", "attacks_strenght"]]
df_component_train = df_component_train[df_component_train.text_id.str.contains(target_dataset)]
df_component_train["tokens"]= df_component_train.component_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())

df_relation_train = pd.read_csv(path+"Minimal/relation_train.csv", header=0)[["text_id", "source_tokens", "target_tokens", "minimalist_labels"]]
df_relation_train = df_relation_train[df_relation_train.text_id.str.contains(target_dataset)]
df_relation_train["t_source_tokens"]= df_relation_train.source_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())
df_relation_train["t_target_tokens"]= df_relation_train.target_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())
df_relation_train.rename(columns={"minimalist_labels": "labels"}, inplace=True)

# Validation
df_component_val = pd.read_csv(path+"component_rel_quality_val.csv", header=0)[["text_id", "component_tokens", "labels", "supports_strenght", "attacks_strenght"]]
df_component_val = df_component_val[df_component_val.text_id.str.contains(target_dataset)]
df_component_val["tokens"]= df_component_val.component_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())

df_relation_val = pd.read_csv(path+"Minimal/relation_val.csv", header=0)[["text_id", "source_tokens", "target_tokens", "minimalist_labels"]]
df_relation_val = df_relation_val[df_relation_val.text_id.str.contains(target_dataset)]
df_relation_val["t_source_tokens"]= df_relation_val.source_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())
df_relation_val["t_target_tokens"]= df_relation_val.target_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())
df_relation_val.rename(columns={"minimalist_labels": "labels"}, inplace=True)

# Test
df_component_test = pd.read_csv(path+"component_rel_quality_test.csv", header=0)[["text_id", "component_tokens", "labels", "supports_strenght", "attacks_strenght"]]
df_component_test["tokens"]= df_component_test.component_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())
df_component_test_argpt = df_component_test[df_component_test.text_id.str.contains("ArGPT")]
df_component_test = df_component_test[df_component_test.text_id.str.contains(target_dataset)]

df_relation_test = pd.read_csv(path+"Minimal/relation_test.csv", header=0)[["text_id", "source_tokens", "target_tokens", "minimalist_labels"]]
df_relation_test["t_source_tokens"]= df_relation_test.source_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())
df_relation_test["t_target_tokens"]= df_relation_test.target_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())
df_relation_test.rename(columns={"minimalist_labels": "labels"}, inplace=True)
df_relation_test_argpt = df_relation_test[df_relation_test.text_id.str.contains("ArGPT")]
df_relation_test = df_relation_test[df_relation_test.text_id.str.contains(target_dataset)]

# Load Components and Relations predicted by the AM Module
df_component_test_pipeline = pd.read_csv(path+"component_results.csv", header=0)
df_component_test_pipeline["tokens"]= df_component_test_pipeline.component_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())
df_component_test_pipeline.rename(columns={"predicted_labels": "labels"}, inplace=True)

df_relation_test_pipeline = pd.read_csv(path+"/relation_results.csv", header=0)
df_relation_test_pipeline["t_source_tokens"]= df_relation_test_pipeline.source_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())
df_relation_test_pipeline["t_target_tokens"]= df_relation_test_pipeline.target_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())
df_relation_test_pipeline.rename(columns={"predicted_labels": "labels"}, inplace=True)


# Join train and val to get more balanced labels
df_component_train_val = pd.concat([df_component_train, df_component_val])
df_component_train_val = df_component_train_val.reset_index(drop=True)
df_component_train_val["id"] = df_component_train_val.index
df_component_train_val = df_component_train_val[df_component_train_val.labels != "Premise"]
df_relation_train_val = pd.concat([df_relation_train, df_relation_val])
df_relation_train_val = df_relation_train_val[df_relation_train_val.labels == "support"]
is_supported = df_relation_train_val["t_target_tokens"].unique()
df_component_train_val = df_component_train_val[df_component_train_val.tokens.isin(is_supported)]

df_component_val_sup = df_component_train_val.groupby(['supports_strenght'], as_index=False).apply(lambda x: x.sample(frac=0.1)).reset_index(drop=True)
ids = df_component_val_sup.id.unique()
df_component_train_sup = df_component_train_val[~df_component_train_val.id.isin(ids)]

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-large", add_prefix_space=True,  use_fast = False)


if num_labels == 5:
  prob_to_labels = {
      "0.0": 0,
      "0.4": 1,
      "0.6": 2,
      "0.8": 3,
      "1.0": 4
  }
  labels_to_prob = {
    0: "0.0",
    1: "0.4",
    2: "0.6",
    3: "0.8",
    4: "1.0"
}
elif num_labels == 3:
  prob_to_labels = {
      "0.0": 0,
      "0.4": 1,
      "0.6": 1,
      "0.8": 2,
      "1.0": 2
  }
  labels_to_prob = {
    0: "0.0",
    1: "0.5",
    2: "1.0"
}
else:
  prob_to_labels = {
      "0.0": 0,
      "0.4": 0,
      "0.6": 1,
      "0.8": 1,
      "1.0": 1
  }
  labels_to_prob = {
    0: "0.0",
    1: "1.0",
}
  

def set_seed(seed_value=42):
  os.environ['PYTHONHASHSEED']=str(seed_value)
  random.seed(seed_value)
  np.random.seed(seed_value)
  tf.compat.v1.set_random_seed(seed_value)
  keras.utils.set_random_seed(seed_value)
  tf.config.experimental.enable_op_determinism()
  

# Preprocessing

def create_class_weight(num_outputs, total, label_0, label_1, label_2, label_3, label_4):
  if num_outputs == 5:
    class_weight = {
        0: round(total/label_0) if label_0 else 1.0,
        1: round(total/label_1) if label_1 else 1.0,
        2: round(total/label_2) if label_2 else 1.0,
        3: round(total/label_3) if label_3 else 1.0,
        4: round(total/label_4) if label_4 else 1.0
    }
  elif num_outputs == 3:
    class_weight = {
        0: round(total/label_0) if label_0 else 1.0,
        1: round(total/label_1) if label_1 else 1.0,
        2: round(total/label_2) if label_2 else 1.0,
    }
  else:
    class_weight = {
        0: round(total/label_0) if label_0 else 1.0,
        1: round(total/label_1) if label_1 else 1.0,
    }
  return class_weight


def prepare_data_rf(df_component, df_relation, prob_to_labels, tokenizer=tokenizer, to_predict=False):

  inputs = list()
  outputs = list()
  list_of_components = list()
  label_0 = 0
  label_1 = 0
  label_2 = 0
  label_3 = 0
  label_4 = 0

  components = df_component.to_dict('records')
  df_r = df_relation[df_relation.labels == "support"]
  for component in components:
    text_id = component["text_id"]
    label = component["labels"]

    if label == "Premise" and "AAEC" in text_id:
      continue

    df_rc = df_r[(df_r.t_target_tokens == component["tokens"]) & (df_r.text_id == text_id)]

    if not to_predict:
      quality = str(component["supports_strenght"])
    else:
      quality = "1.0"
    if quality == "0.48":
      continue
    all_source_tokens = df_rc.source_tokens.unique()
    to_be_tokenized = f"</s> {component['component_tokens']} </s>"
    for source in all_source_tokens:
      to_be_tokenized = f"{to_be_tokenized}</s> {source} </s>"
    token = tokenizer.encode(to_be_tokenized, padding="max_length", max_length=333)
    if len(token) > 333:
      print(len(token))
    inputs.append(token)
    outputs.append(prob_to_labels[quality])
    if quality == "0.0":
      label_0 = label_0 + 1
    elif quality == "0.4":
      label_1 = label_1 + 1
    elif quality == "0.6":
      label_2 = label_2 + 1
    elif quality == "0.8":
      label_3 = label_3 + 1
    else:
      label_4 = label_4 + 1
    list_of_components.append({
        "text_id": text_id,
        "component_tokens": component['component_tokens'],
        "labels": label
    })
  np_inputs = np.asarray(inputs, dtype=np.float32)
  np_outputs = np.asarray(outputs, dtype=np.float32)

  total = label_0 + label_1 + label_2 + label_3 + label_4
  num_outputs = len(list(set(prob_to_labels.values())))
  class_weight = create_class_weight(num_outputs, total, label_0, label_1, label_2, label_3, label_4)

  return np_inputs, np_outputs, class_weight, list_of_components


# Preprocess data
X_train, y_train, class_weight, _ = prepare_data_rf(df_component_train, df_relation_train_val, prob_to_labels)
X_test, y_test, _, list_of_test_components = prepare_data_rf(df_component_test, df_relation_test, prob_to_labels)

# Prepare test set for ArGPT (original components)
X_test_gpt, y_test_gpt, _, new_components_gpt = prepare_data_rf(df_component_test_argpt, df_relation_test_argpt, prob_to_labels, to_predict=True)
# Prepare test set for components generated by the AM Module
X_test_pipeline, y_test_pipeline, _, components_pipeline = prepare_data_rf(df_component_test_pipeline, df_relation_test_pipeline, prob_to_labels, to_predict=True)

micro = list()
macro = list()
weighted = list()
seeds = [0, 27, 42, 1827, 23, 2000, 11, 9, 1956, 15]

for seed in seeds:
  set_seed(seed)
  rf = RandomForestClassifier(class_weight=class_weight, n_estimators=300, random_state=seed)
  rf.fit(X_train, y_train)
  y_pred = rf.predict(X_test)
  f1_micro = f1_score(average="micro", y_true=y_test, y_pred=y_pred)
  f1_macro = f1_score(average="macro", y_true=y_test, y_pred=y_pred)
  f1_weighted = f1_score(average="weighted", y_true=y_test, y_pred=y_pred)
  print(f"SEED: {seed} -> F1 Micro: {f1_micro}, F1 Macro: {f1_macro}, F1 Weighted: {f1_weighted}")

  micro.append(f1_micro)
  macro.append(f1_macro)
  weighted.append(f1_weighted)

micro_mean = np.mean(micro)
macro_mean = np.mean(macro)
weighted_mean = np.mean(weighted)
print(f"\n F1 Micro: {micro_mean}, F1 Macro: {macro_mean}, F1 Weighted: {weighted_mean}")


# Generate output files with predictions
rf = RandomForestClassifier(class_weight=class_weight, n_estimators=300)
rf.fit(X_train, y_train)

# For the original AAEC components
y_pred = rf.predict(X_test)
original_components_predicted_support_quality = list()

for num, component in enumerate(list_of_test_components):
  original_components_predicted_support_quality.append({
        "text_id": component["text_id"],
        "component_tokens": component['component_tokens'],
        "labels": component["labels"],
        "predicted_supports_quality": labels_to_prob[int(y_pred[num])]
    })

df_original_components_predicted_support_quality = pd.DataFrame.from_dict(original_components_predicted_support_quality)

# For the original ArGPT components
y_pred = rf.predict(X_test_gpt)
original_components_predicted_support_quality_argpt = list()

for num, component in enumerate(new_components_gpt):
  original_components_predicted_support_quality_argpt.append({
        "text_id": component["text_id"],
        "component_tokens": component['component_tokens'],
        "labels": component["labels"],
        "predicted_supports_quality": labels_to_prob[int(y_pred[num])]
    })

df_original_components_predicted_support_quality_argpt = pd.DataFrame.from_dict(original_components_predicted_support_quality_argpt)

# For predicted components
y_pred = rf.predict(X_test_pipeline)
pipeline_components_predicted_support_quality = list()

for num, component in enumerate(components_pipeline):
  pipeline_components_predicted_support_quality.append({
        "text_id": component["text_id"],
        "component_tokens": component['component_tokens'],
        "labels": component["labels"],
        "predicted_supports_quality": labels_to_prob[int(y_pred[num])]
    })

df_pipeline_components_predicted_support_quality = pd.DataFrame.from_dict(pipeline_components_predicted_support_quality)
