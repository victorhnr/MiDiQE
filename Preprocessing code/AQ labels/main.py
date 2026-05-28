import os
import re
import pandas as pd

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

# Load AAEC dataset
path = "./dataset"

target_dataset = "AAEC"

# Train
df_component_train = pd.read_csv(path+"component_train.csv", header=0)[["text_id", "component_tokens", "labels"]]
df_component_train = df_component_train[df_component_train.text_id.str.contains(target_dataset)]
df_component_train["tokens"]= df_component_train.component_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())

df_relation_train = pd.read_csv(path+"relation_train.csv", header=0)[["text_id", "source_tokens", "target_tokens", "labels"]]
df_relation_train = df_relation_train[df_relation_train.text_id.str.contains(target_dataset)]
df_relation_train["target_tokens"]= df_relation_train.target_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())

#Validation
df_component_val = pd.read_csv(path+"component_val.csv", header=0)[["text_id", "component_tokens", "labels"]]
df_component_val = df_component_val[df_component_val.text_id.str.contains(target_dataset)]
df_component_val["tokens"]= df_component_val.component_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())

df_relation_val = pd.read_csv(path+"relation_val.csv", header=0)[["text_id", "source_tokens", "target_tokens", "labels"]]
df_relation_val = df_relation_val[df_relation_val.text_id.str.contains(target_dataset)]
df_relation_val["source_tokens"]= df_relation_val.source_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())
df_relation_val["target_tokens"]= df_relation_val.target_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())

#Test
df_component_test = pd.read_csv(path+"component_test.csv", header=0)[["text_id", "component_tokens", "labels"]]
df_component_test = df_component_test[df_component_test.text_id.str.contains(target_dataset)]
df_component_test["tokens"]= df_component_test.component_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())

df_relation_test = pd.read_csv(path+"relation_test.csv", header=0)[["text_id", "source_tokens", "target_tokens", "labels"]]
df_relation_test = df_relation_test[df_relation_test.text_id.str.contains(target_dataset)]
df_relation_test["source_tokens"]= df_relation_test.source_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())
df_relation_test["target_tokens"]= df_relation_test.target_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())


data = {
    "train": {
        "component": df_component_train,
        "relation": df_relation_train
    },
    "test": {
        "component": df_component_test,
        "relation": df_relation_test
    },
    "val": {
        "component": df_component_val,
        "relation": df_relation_val
    },
}

text_components = list()

score_to_prob = {
    "0": 0, "10": 40, "15": 60, "20": 80, "25": 100
}

for key in data.keys():
  text_ids = data[key]["relation"].text_id.unique()
  comp = list()
  for text_id in text_ids:
    if text_id in ["AAEC_212", "AAEC_213"]:
      continue
    text_comp_num = 0
    essay_number =  int(text_id.replace("AAEC_", ""))
    df_comp =  new_df[(new_df.essay_number == essay_number) & (new_df.component_type == "MajorClaim") ]
    text_arguments = df_comp.to_dict("records")
    waiting_score = dict()
    for argument in text_arguments:
      argument_id = re.sub(r'[^\w]', ' ', argument["component_text"]).strip().replace(" ", "").lower()
      congency = int(argument["cogency_score"].replace("Cogency_", "").replace("cannotJudge", "15"))
      supports_strenght = score_to_prob[str(congency)]/100
      if type(argument["reasonableness_counterargument"]) != float:
        reasonableness_counterargument = int(argument["reasonableness_counterargument"].replace("Reasonableness_counterargument_", "").replace("cannotJudge", "15"))
        attacks_strenght = score_to_prob[str(reasonableness_counterargument)]/100
      else:
        attacks_strenght = 1.0
      if type(argument["reasonableness_rebuttal"]) != float:
        reasonableness_rebuttal = int(argument["reasonableness_rebuttal"].replace("Reasonableness_rebuttal_", "").replace("cannotJudge", "15"))
        df_rel = data[key]["relation"]
        df_r = df_rel[(df_rel.minimalist_labels == "attack") & (df_rel.text_id == text_id) & (df_rel.target_tokens == argument_id)]
        source_id = df_r.source_tokens.unique()[0]
        waiting_score[source_id] = score_to_prob[str(reasonableness_rebuttal)]/100
      df =  data[key]["component"]
      df = df[df.text_id == text_id]
      df = df[df.labels == "MajorClaim"]
      text_mcs = df.to_dict("records")
      for text_mc in text_mcs:
        text_components.append({
            "text_id": text_id,
            "component_tokens": text_mc["component_tokens"],
            "labels": argument["component_type"],
            "supports_strenght": supports_strenght,
            "attacks_strenght": attacks_strenght,
        })
        text_comp_num = text_comp_num + 1

    df_comp =  new_df[(new_df.essay_number == essay_number) & (new_df.component_type == "Claim") ]
    text_arguments = df_comp.to_dict("records")
    for argument in text_arguments:
      argument_id = re.sub(r'[^\w]', ' ', argument["component_text"]).strip().replace(" ", "").lower()
      congency = int(argument["cogency_score"].replace("Cogency_", "").replace("cannotJudge", "15"))
      supports_strenght = score_to_prob[str(congency)]/100
      if type(argument["reasonableness_counterargument"]) != float:
        reasonableness_counterargument = int(argument["reasonableness_counterargument"].replace("Reasonableness_counterargument_", "").replace("cannotJudge", "15"))
        attacks_strenght = score_to_prob[str(reasonableness_counterargument)]/100
      else:
        try:
          attacks_strenght = waiting_score[argument_id]
        except:
          attacks_strenght = 1.0
      if type(argument["reasonableness_rebuttal"]) != float:
        reasonableness_rebuttal = int(argument["reasonableness_rebuttal"].replace("Reasonableness_rebuttal_", "").replace("cannotJudge", "15"))
        df_rel = data[key]["relation"]
        df_r = df_rel[(df_rel.labels == "attack") & (df_rel.text_id == text_id) & (df_rel.target_tokens == argument_id)]
        if not df_r.empty:
          source_id = df_r.source_tokens.unique()[0]
          waiting_score[source_id] = score_to_prob[str(reasonableness_rebuttal)]/100
      text_components.append({
          "text_id": text_id,
          "component_tokens": argument["component_text"],
          "labels": argument["component_type"],
          "supports_strenght": supports_strenght,
          "attacks_strenght": attacks_strenght,
      })
      text_comp_num = text_comp_num + 1
    df_comp =  data[key]["component"]
    df_comp = df_comp[df_comp.text_id == text_id]
    num = df_comp.shape[0]
    df_comp = df_comp[df_comp.labels == "Premise"]
    text_arguments = df_comp.to_dict("records")

    for argument in text_arguments:
      argument_id = argument["tokens"]
      try:
        attacks_strenght = waiting_score[argument_id]
      except:
        attacks_strenght = 1.0
      text_components.append({
            "text_id": text_id,
            "component_tokens": argument["component_tokens"],
            "labels": argument["labels"],
            "supports_strenght": 1.0,
            "attacks_strenght": attacks_strenght,
        })
      text_comp_num = text_comp_num + 1

    if text_comp_num != num:
      print(text_id)

  df_comp = data[key]["component_argpt"]
  text_arguments = df_comp.to_dict("records")

  for argument in text_arguments:
    text_components.append({
            "text_id": argument["text_id"],
            "component_tokens": argument["component_tokens"],
            "labels": argument["minimalist_labels"],
            "supports_strenght": 1.0,
            "attacks_strenght": 1.0,
        })

df_final = pd.DataFrame(text_components, columns = text_components[0].keys())
df_final.to_csv(f"{path}/component_rel_quality_val.csv")
