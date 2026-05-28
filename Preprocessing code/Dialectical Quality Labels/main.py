import os
import re
import pandas as pd
import numpy as np
import math
from statistics import mean


path = "./persuasive-essays-argument-quality-dataset-main/Arg_Quality_dataset/data"

df_quality = pd.read_csv(path+"/quality_annotation_table.csv", sep="\t", header=0)

df_quality.rename(columns={"essay_number ": "essay_number"}, inplace=True)
df_quality.rename(columns={" component_id ": "component_id"}, inplace=True)
df_quality.rename(columns={" cogency_score ": "cogency_score"}, inplace=True)
df_quality.rename(columns={" rhetorical_strategy ": "rhetorical_strategy"}, inplace=True)
df_quality.rename(columns={" reasonableness_counterargument ": "reasonableness_counterargument"}, inplace=True)
df_quality.rename(columns={" reasonableness_rebuttal": "reasonableness_rebuttal"}, inplace=True)
df_quality = df_quality.drop_duplicates()

path = "./persuasive-essays-argument-quality-dataset-main/Arg_Quality_dataset/data/annotations"

directory_files = os.listdir(path)
abstracts_list = list()
for dfile in directory_files:
  file_name = dfile.replace(".ann", "").replace(".txt", "")
  if file_name not in abstracts_list:
    abstracts_list.append(file_name)

abstracts_list = list(set(abstracts_list))

components = list()

for i, file_id in enumerate(abstracts_list):
  print(f"{i+1} out of {len(abstracts_list)}")
  arg_path = f"{path}/{file_id}.txt"
  raw_text = open(arg_path, "r", encoding="utf-8").read()
  text_lines = raw_text.split("\n")
  _ ,component_type, component_id = text_lines[4].split(" ")
  component_text = text_lines[5].split(": ")[1]
  text_id = int(file_id.replace("essay","").split("_")[0])

  components.append({
      "essay_number": text_id,
      "component_id": component_id,
      "component_type": component_type,
      "component_text": component_text
  })
df_components = pd.DataFrame(components, columns = components[0].keys())
df_components = df_components.drop_duplicates()

new_df = pd.merge(df_quality, df_components,  how='left', left_on=['essay_number','component_id'], right_on = ['essay_number','component_id'])

# Preprocess Dialectical Quality for AAEC

file_path = "./"
target_dataset = "AAEC"

df_component_train = pd.read_csv(path+"component_train.csv", header=0)[["text_id", "component_tokens", "labels", "minimalist_labels"]]
df_component_train_argpt = df_component_train[df_component_train.text_id.str.contains("ArGPT")]
df_component_train = df_component_train[df_component_train.text_id.str.contains(target_dataset)]
df_component_train["tokens"]= df_component_train.component_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())

df_relation_train = pd.read_csv(path+"relation_train.csv", header=0)[["text_id", "source_tokens", "target_tokens", "labels", "minimalist_labels"]]
df_relation_train = df_relation_train[df_relation_train.text_id.str.contains(target_dataset)]
df_relation_train["source_tokens"]= df_relation_train.source_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())
df_relation_train["target_tokens"]= df_relation_train.target_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())


df_component_val = pd.read_csv(path+"component_val.csv", header=0)[["text_id", "component_tokens", "labels", "minimalist_labels"]]
df_component_val_argpt = df_component_val[df_component_val.text_id.str.contains("ArGPT")]
df_component_val = df_component_val[df_component_val.text_id.str.contains(target_dataset)]
df_component_val["tokens"]= df_component_val.component_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())

df_relation_val = pd.read_csv(path+"relation_val.csv", header=0)[["text_id", "source_tokens", "target_tokens", "labels", "minimalist_labels"]]
df_relation_val = df_relation_val[df_relation_val.text_id.str.contains(target_dataset)]
df_relation_val["source_tokens"]= df_relation_val.source_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())
df_relation_val["target_tokens"]= df_relation_val.target_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())


df_component_test = pd.read_csv(path+"component_test.csv", header=0)[["text_id", "component_tokens", "labels", "minimalist_labels"]]
df_component_test_argpt = df_component_test[df_component_test.text_id.str.contains("ArGPT")]
df_component_test = df_component_test[df_component_test.text_id.str.contains(target_dataset)]
df_component_test["tokens"]= df_component_test.component_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())

df_relation_test = pd.read_csv(path+"relation_test.csv", header=0)[["text_id", "source_tokens", "target_tokens", "labels", "minimalist_labels"]]
df_relation_test = df_relation_test[df_relation_test.text_id.str.contains(target_dataset)]
df_relation_test["source_tokens"]= df_relation_test.source_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())
df_relation_test["target_tokens"]= df_relation_test.target_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())

df_relations =  pd.concat([df_relation_train, df_relation_val])
df_relations =  pd.concat([df_relations, df_relation_test])

# For text quality

essay_quality = list()
text_ids = new_df.essay_number.unique()

score_to_prob = {
    "0": 0, "10": 40, "15": 60, "20": 80, "25": 100
}

for text_id in text_ids:
  major_claim_strengths = list()
  claim_strenght = list()
  text_df = new_df[new_df.essay_number == text_id]
  text_relations_df = df_relations[df_relations.text_id == f"AAEC_{text_id:03}"]
  arguments = text_df.to_dict("records")
  for argument in arguments:
    if argument["component_type"] == "Claim":
      component_text = argument["component_text"]
      text_rel_df = text_relations_df[(text_relations_df.source_tokens == component_text) & (text_relations_df.minimalist_labels == "support")]
      if text_rel_df.shape[0] != 0:
        c_cogency = int(argument["cogency_score"].replace("Cogency_", "").replace("cannotJudge", "15"))
        c_congency_prob = score_to_prob[str(c_cogency)]
        claim_strenght.append(c_congency_prob)
  if len(claim_strenght) == 0:
    claim_strenght = [100]
  claims_cogency = mean(claim_strenght)/100

  current_strength = 0
  for argument in arguments:
    if argument["component_type"] != "MajorClaim":
      continue
    else:
      strenght = list()
      congency = int(argument["cogency_score"].replace("Cogency_", "").replace("cannotJudge", "15"))
      congency_prob = score_to_prob[str(congency)]
      corrected_cogency = claims_cogency*congency_prob
      prob_counterargument = 0
      if type(argument["reasonableness_counterargument"]) != float:
        reasonableness_counterargument = int(argument["reasonableness_counterargument"].replace("Reasonableness_counterargument_", "").replace("cannotJudge", "15"))
        prob_counterargument = score_to_prob[str(reasonableness_counterargument)]
      prob_rebuttal = 0
      if type(argument["reasonableness_rebuttal"]) != float:
        reasonableness_rebuttal = int(argument["reasonableness_rebuttal"].replace("Reasonableness_rebuttal_", "").replace("cannotJudge", "15"))
        prob_rebuttal = score_to_prob[str(reasonableness_rebuttal)]
      if prob_counterargument == 0 and prob_rebuttal == 0:
        current_strength = corrected_cogency
      elif prob_counterargument > prob_rebuttal:
        current_strength = min(max(corrected_cogency - prob_counterargument + prob_rebuttal, 0), 100)
      else:
        current_strength = (corrected_cogency + prob_counterargument + prob_rebuttal)/3

      major_claim_strengths.append(current_strength)
  arg_score = math.ceil(mean(major_claim_strengths))
  if arg_score > 75:
    arg_quality = "good"
    arg_quality_fine_grained = "good"
  elif arg_score > 50:
    arg_quality = "good"
    arg_quality_fine_grained = "mostly good"
  elif arg_score > 25:
    arg_quality = "bad"
    arg_quality_fine_grained = "mostly bad"
  else:
    arg_quality = "bad"
    arg_quality_fine_grained = "bad"
  essay_quality.append({
      "text_id": text_id,
      "arg_score": arg_score,
      "arg_quality_fine_grained": arg_quality_fine_grained,
      "arg_quality": arg_quality,
  })

df_essay_quality = pd.DataFrame(essay_quality, columns = essay_quality[0].keys())

df_quality_train = pd.merge(df_component_train, df_essay_quality,  how='left', left_on=['text_id'], right_on = ['text_id'])
df_quality_train = df_quality_train[df_quality_train.arg_quality_fine_grained.notna()]

df_quality_val = pd.merge(df_component_val, df_essay_quality,  how='left', left_on=['text_id'], right_on = ['text_id'])
df_quality_val = df_quality_val[df_quality_val.arg_quality_fine_grained.notna()]

df_quality_test = pd.merge(df_component_test, df_essay_quality,  how='left', left_on=['text_id'], right_on = ['text_id'])
df_quality_test = df_quality_test[df_quality_test.arg_quality_fine_grained.notna()]

df_quality_train["text_id"] = df_quality_train["text_id"].apply(lambda x: f"AAEC_{x}")
df_quality_val["text_id"] = df_quality_val["text_id"].apply(lambda x: f"AAEC_{x}")
df_quality_test["text_id"] = df_quality_test["text_id"].apply(lambda x: f"AAEC_{x}")

# Preprocess for ArGPT
path = "./GPT-dataset/Final"

df_argpt = pd.read_csv(path+"/text_evaluations.csv", header=0)
df_argpt = df_argpt[["text_id", "tokens", "criteria_7", "criteria_8", "criteria_9", "split"]]
df_argpt["criteria_7"] = df_argpt["criteria_7"].apply(lambda x: 4*float(x))
df_argpt["criteria_8"] = df_argpt["criteria_8"].apply(lambda x: 4*float(x))
df_argpt["criteria_9"] = df_argpt["criteria_9"].apply(lambda x: 4*float(x))

gpt_quality = list()
gpt_data = df_argpt.to_dict("records")

for data in gpt_data:
  arg_score = 10*(data["criteria_7"] + 2*data["criteria_8"] + data["criteria_9"])/4
  if arg_score > 50:
    arg_quality = "good"
  else:
    arg_quality = "bad"

  gpt_quality.append({
      "text_id": f"ArGPT_{data['text_id']}",
      "text_tokens": data["tokens"],
      "arg_score": arg_score,
      "arg_quality": arg_quality,
      "split": data["split"]
  })

df_gpt_quality = pd.DataFrame(gpt_quality, columns = gpt_quality[0].keys())
df_gpt_quality_train = df_gpt_quality[df_gpt_quality.split == "TRAIN"][["text_id", "text_tokens", "arg_score", "arg_quality"]]
df_gpt_quality_val = df_gpt_quality[df_gpt_quality.split == "VAL"][["text_id", "text_tokens", "arg_score", "arg_quality"]]
df_gpt_quality_test = df_gpt_quality[df_gpt_quality.split == "TEST"][["text_id", "text_tokens", "arg_score", "arg_quality"]]

# Combine preprocessing of AAEC and ArGPT
df_quality_train = pd.concat([df_quality_train, df_gpt_quality_train], ignore_index=True)
df_quality_val = pd.concat([df_gpt_quality_val, df_quality_val], ignore_index=True)
df_quality_test = pd.concat([df_quality_test, df_gpt_quality_test], ignore_index=True)

df_quality_train.to_csv(f"{file_path}/arg_quality_train.csv")
df_quality_val.to_csv(f"{file_path}/arg_quality_val.csv")
df_quality_test.to_csv(f"{file_path}/arg_quality_test.csv")
