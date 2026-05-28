import os
import re
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from transformers import AutoTokenizer
from tensorflow.keras.models import Model
from tensorflow.keras import layers, metrics


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
      "0.0": [1, 0, 0, 0, 0],
      "0.4": [0, 1, 0, 0, 0],
      "0.6": [0, 0, 1, 0, 0],
      "0.8": [0, 0, 0, 1, 0],
      "1.0": [0, 0, 0, 0, 1]
  }
elif num_labels == 3:
  prob_to_labels = {
      "0.0": [1, 0, 0],
      "0.4": [0, 1, 0],
      "0.6": [0, 1, 0],
      "0.8": [0, 0, 1],
      "1.0": [0, 0, 1]
  }
else:
  prob_to_labels = {
      "0.0": [1, 0],
      "0.4": [1, 0],
      "0.6": [0, 1],
      "0.8": [0, 1],
      "1.0": [0, 1]
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

# Preprocess Data
def prepare_data_support(df_component, df_relation, prob_to_labels, tokenizer=tokenizer, to_predict=False):

  inputs_s = list()
  inputs_t = list()
  outputs = list()
  label_0 = 0
  label_1 = 0
  label_2 = 0
  label_3 = 0
  label_4 = 0

  components = df_component.to_dict('records')
  new_components = list()
  df_r = df_relation[df_relation.labels == "support"]
  for component in components:
    text_id = component["text_id"]
    label = component["labels"]
    if label == "Premise" and not to_predict:
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
    to_be_tokenized_sources = ""
    for source in all_source_tokens:
      to_be_tokenized_sources = f"{to_be_tokenized_sources}</s> {source} </s>"
    to_be_tokenized_sources = f"{to_be_tokenized_sources}</s>"
    token = tokenizer.encode(to_be_tokenized, padding="max_length", max_length=63)
    token_sources = tokenizer.encode(to_be_tokenized_sources, padding="max_length", max_length=230)
    if len(token) > 63:
      print(len(token))
    if len(token_sources) > 230:
      print("Sources")
      print(len(token_sources))
    inputs_t.append(token)
    inputs_s.append(token_sources)
    try:
      outputs.append(prob_to_labels[quality])
    except:
      print(quality)
      print(component)
      raise

    if quality in ("0.0"):
      label_0 = label_0 + 1
    elif quality in ("0.4"):
      label_1 = label_1 + 1
    elif quality in ("0.6"):
      label_2 = label_2 + 1
    elif quality == "0.8":
      label_3 = label_3 + 1
    else:
      label_4 = label_4 + 1
    new_components.append(component)
  np_inputs_target = np.asarray(inputs_t, dtype=np.float32)
  np_inputs_source = np.asarray(inputs_s, dtype=np.float32)
  np_outputs = np.asarray(outputs, dtype=np.float32)
  total = label_0 + label_1 + label_2 + label_3 + label_4

  num_outputs = len(list(prob_to_labels['0.0']))
  class_weight = create_class_weight(num_outputs, total, label_0, label_1, label_2, label_3, label_4)

  return np_inputs_target, np_inputs_source, np_outputs, class_weight, new_components


# Create LSTM Cogency Model

def create_model(num_outputs, act="selu"):

  input_target = layers.Input(shape=(63, 1))
  x_target = layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.0001)(input_target)
  x_target = layers.Bidirectional(layers.LSTM(64))(input_target)
  x_target = layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.0001)(x_target)
  x_target = layers.Dropout(0.2)(x_target)
  x_target = layers.Dense(50, activation=act)(x_target)
  x_target = layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.0001)(x_target)
  x_target = layers.Dropout(0.2)(x_target)
  x_target = layers.Flatten()(x_target)

  input_source = layers.Input(shape=(230, 1))
  x_source = layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.0001)(input_source)
  x_source = layers.Bidirectional(layers.LSTM(128))(input_source)
  x_source = layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.0001)(x_source)
  x_source = layers.Dropout(0.2)(x_source)
  x_source = layers.Dense(100, activation=act)(x_source)
  x_source = layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.0001)(x_source)
  x_source = layers.Dropout(0.2)(x_source)
  x_source = layers.Flatten()(x_source)

  merge = layers.concatenate([x_source, x_target])
  merge = layers.Reshape((150, 1))(merge)
  merge = layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.0001)(merge)
  x = layers.Bidirectional(layers.LSTM(128))(merge)
  x = layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.0001)(x)
  x = layers.Dropout(0.2)(x)
  x = layers.Dense(75, activation=act)(x)
  x = layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.0001)(x)
  x = layers.Dropout(0.2)(x)
  x = layers.Dense(25, activation=act)(x)
  x = layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.0001)(x)
  x = layers.Dropout(0.2)(x)
  y = layers.Dense(num_outputs, activation="softmax")(x)

  model = Model(inputs=[input_target, input_source], outputs=y)

  model.compile(
      loss=keras.losses.CategoricalCrossentropy(),
      metrics=[
          metrics.F1Score(average="micro", name="f1_micro"),
          metrics.F1Score(average="macro", name="f1_macro"),
          metrics.F1Score(average="weighted", name="f1_weighted"),
      ],
      optimizer="adam"
  )
  return model

# Train LSTM Model

def train_model_with_seed(model, seed_value, X_train_t, X_train_s, y_train, X_val_t, X_val_s, y_val, class_weight, X_test_t, X_test_s, y_test, num_outputs):
  task = "Cogency"
  set_seed(seed_value)
  # define the checkpoint
  filepath=f"./Arg_Quality_tests/Seed_{seed_value}/{task}_{num_outputs}_labels.keras"
  checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_f1_macro', verbose=1, save_best_only=True, mode="max")
  lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_f1_macro',
    patience=6,
    factor=0.5
  )
  callbacks_list = [checkpoint, lr]
  #Train Model
  model.fit([X_train_t, X_train_s], y_train, validation_data=([X_val_t, X_val_s], y_val), callbacks=callbacks_list, shuffle=True, epochs=30, batch_size=32, class_weight=class_weight)
  model.load_weights(filepath)
  _, f1_micro, f1_macro, f1_weighted = model.evaluate([X_test_t, X_test_s], y_test)

  print(f"\n\n SEED {seed_value} --> F1 Micro: {f1_micro}, F1 Macro: {f1_macro}, F1 Weighted: {f1_weighted} \n\n")
  return f1_micro, f1_macro, f1_weighted

# Preprocess data
# Prepare data for training and validation
X_train_t, X_train_s, y_train, class_weight, _ = prepare_data_support(df_component_train_sup, df_relation_train_val, prob_to_labels=prob_to_labels)
X_val_t, X_val_s, y_val, val_weight, _ = prepare_data_support(df_component_val_sup, df_relation_train_val, prob_to_labels=prob_to_labels)
# Prepare test set on the original components from AAEC
X_test_t, X_test_s, y_test, test_weight, new_components= prepare_data_support(df_component_test, df_relation_test, prob_to_labels=prob_to_labels)
# Prepare test set for ArGPT (original components)
X_test_t_gpt, X_test_s_gpt, _, _, new_components_gpt = prepare_data_support(df_component_test_argpt, df_relation_test_argpt, prob_to_labels=prob_to_labels, to_predict=True)
# Prepare test set for components generated by the AM Module
X_test_t_pipeline, X_test_s_pipeline, _, _, components_pipeline = prepare_data_support(df_component_test_pipeline, df_relation_test_pipeline, prob_to_labels=prob_to_labels, to_predict=True)


micro = list()
macro = list()
weighted = list()
seeds = [0, 27, 42, 1827, 23, 2000, 11, 9, 1956, 15]

for seed in seeds:
  print(f"SEED: {seed}")
  model = create_model(num_labels)
  f1_micro, f1_macro, f1_weighted = train_model_with_seed(model, seed, X_train_t, X_train_s, y_train, X_val_t, X_val_s, y_val, class_weight, X_test_t, X_test_s, y_test, num_labels)
  micro.append(f1_micro)
  macro.append(f1_macro)
  weighted.append(f1_weighted)

micro_mean = np.mean(micro)
macro_mean = np.mean(macro)
weighted_mean = np.mean(weighted)

print(f"F1 Micro: {micro_mean}, F1 Macro: {macro_mean}, F1 Weighted: {weighted_mean}")
