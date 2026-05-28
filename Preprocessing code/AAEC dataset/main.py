import pandas as pd
import re
from brat_parser import get_entities_relations_attributes_groups

split_file = "./AAEC/original data/train-test-split.csv"
df_split = pd.read_csv(split_file, sep=";", names=["ID", "SET"], header=0)
data_split = dict(df_split.values)

# Load and preprocess data
# Dataset I - Text and TokenClassification
# Dataset II - Text, Component, Label
# Dataset III - Text, Source Component, Target Component, labels

def intersects(interval, matches_found):
  start = interval[0]
  end = interval[-1]
  for i, match_found in enumerate(matches_found):
    found_start = match_found[0]
    found_end = match_found[-1]
    # Intersection
    if max(0, min(end, found_end) - max(start, found_start)) > 0:
      return True, i

  return False, None



def subfinder(mylist, key, num, matches_found, entities):
    element = entities[key].text.strip()
    pattern = re.findall(r"[\w']+|[.,!?;]", element)
    matches = list()
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
          matches.append(list(range(i, i+len(pattern))))

    for match in matches:
        do_intersect, element = intersects(match, matches_found)
        if do_intersect and len(matches) > 1:
          continue
        elif not do_intersect:
          matches_found[num] = match
          break
        elif do_intersect and len(matches) == 1:
            matches_found[element] = [0, 0]
            matches_found[num] = match
            matches_found = subfinder(
                mylist, entities.key()[element], element, matches_found, entities
            )
            break

    return matches_found


def create_text_lists(text_id, entities, raw_text):
  # Load text and turn into a list
  text_list = re.findall(r"[\w']+|[.,!?;]", raw_text.strip())

  # Create annotaded list
  matches = list()
  for i in range(len(entities.keys())):
    matches.append([0, 0])

  for num, key in enumerate(entities.keys()):
    matches = subfinder(text_list, key, num, matches, entities)

  an_text = [0]*len(text_list)
  for match in matches:
    for n in match:
      an_text[n] = 2
    an_text[match[0]] = 1
  return [text_id, text_list, an_text]


train_data_span = list()
test_data_span = list()
train_data_component = list()
test_data_component = list()
train_data_relation = list()
test_data_relation = list()
path = ""

for essay in data_split.keys():

  essay_an_path = path + essay + ".ann"
  essay_path = path + essay + ".txt"
  text_num = re.findall(r'\d+', essay)
  text_id = text_num[0]
  print(text_id)
  text_id = "AAEC_" + text_id
  # Load Annotations
  entities, relations, attributes, groups = get_entities_relations_attributes_groups(essay_an_path)
  # Load text and turn into a list
  raw_text = open(essay_path, "r", encoding="utf-8").read()
  raw_text = raw_text.replace("\n", " ")
  text_lists = create_text_lists(text_id, entities, raw_text)
  # Save to train or test span sets
  if data_split[essay] == "TRAIN":
    train_data_span.append(text_lists)
  else:
    test_data_span.append(text_lists)

  # Save to train or test component sets
  major_claim = None
  m_claims = list()
  for key in entities.keys():
    element = entities[key].text
    component = entities[key].type
    if component == "MajorClaim":
      m_claims.append(key)
    component_list = [text_id, element, raw_text, component]
    if data_split[essay] == "TRAIN":
      train_data_component.append(component_list)
    else:
      test_data_component.append(component_list)

  for key in relations.keys():
    row = relations[key]
    relation = row.type
    source_id = row.subj
    target_id = row.obj
    source_text = entities[source_id].text
    target_text = entities[target_id].text
    relation_list = [text_id, source_id, source_text, target_id, target_text, relation]
    if data_split[essay] == "TRAIN":
      train_data_relation.append(relation_list)
    else:
      test_data_relation.append(relation_list)

  for key in attributes.keys():
    attribute = attributes[key]
    att_type = attribute.type
    if att_type == "Stance":
      source_id = attribute.target
      source_text = entities[source_id].text
      relation = attribute.values[0]
      if relation == "For":
        relation = "support"
      else:
        relation = "attack"

      for major_claim in m_claims:
        target_id = major_claim
        target_text = entities[target_id].text
        relation_list = [text_id, source_id, source_text, target_id,target_text, relation]
        if data_split[essay] == "TRAIN":
          train_data_relation.append(relation_list)
        else:
          test_data_relation.append(relation_list)

  if data_split[essay] == "TRAIN":
    df_rel = pd.DataFrame(train_data_relation, columns = ["text_id", "source_id", "source_tokens", "target_id", "target_tokens", "labels"])
  else:
    df_rel = pd.DataFrame(test_data_relation, columns = ["text_id", "source_id", "source_tokens", "target_id", "target_tokens", "labels"])

  for key in entities.keys():
    source_id = entities[key].id
    source_text = entities[key].text
    relation = "None"
    for new_key in entities.keys():
      target_id = entities[new_key].id
      target_text = entities[new_key].text
      relation_list = [text_id, source_id, source_text, target_id, target_text, relation]
      df_present = df_rel[(df_rel["text_id"] == text_id) & (df_rel["source_id"] == source_id) & (df_rel["target_id"] == target_id)]

      if df_present.shape[0] == 0:
        if data_split[essay] == "TRAIN":
          train_data_relation.append(relation_list)
        else:
          test_data_relation.append(relation_list)

df_span_train = pd.DataFrame(train_data_span, columns = ["text_id", "tokens", "chunk_tags"])
df_span_test = pd.DataFrame(test_data_span, columns = ["text_id", "tokens", "chunk_tags"])
df_component_train = pd.DataFrame(train_data_component, columns = ["text_id", "component_tokens", "text_tokens", "labels", "minimalist_labels"])
df_component_test = pd.DataFrame(test_data_component, columns = ["text_id", "component_tokens", "text_tokens", "labels", "minimalist_labels"])
df_relation_train = pd.DataFrame(train_data_relation, columns = ["text_id", "source_id", "source_tokens", "target_id", "target_tokens", "labels", "minimalist_labels"])
df_relation_test = pd.DataFrame(test_data_relation, columns =  ["text_id", "source_id", "source_tokens", "target_id", "target_tokens", "labels", "minimalist_labels"])

# Randomly create Train, Val and Test sets

# Span
df_span_val = df_span_train.sample(frac=0.10, random_state=200)
val_text_ids = df_span_val["text_id"].to_list()
df_span_val = df_span_train.loc[df_span_train["text_id"].isin(val_text_ids)]
df_span_train = df_span_train.loc[~df_span_train["text_id"].isin(val_text_ids)]
test_text_ids = df_span_train["text_id"].to_list()

# Component
df_component_val = df_component_train.loc[df_component_train["text_id"].isin(val_text_ids)]
df_component_train = df_component_train.loc[df_component_train["text_id"].isin(test_text_ids)]

# Relation
df_relation_val = df_relation_train.loc[df_relation_train["text_id"].isin(val_text_ids)]
df_relation_train = df_relation_train.loc[df_relation_train["text_id"].isin(test_text_ids)]

save_path = "./AAEC/New Preprocessed"

df_span_train.to_csv(f"{save_path}/span_train.csv")
df_span_test.to_csv(f"{save_path}/span_test.csv")
df_span_val.to_csv(f"{save_path}/span_val.csv")
df_component_train.to_csv(f"{save_path}/component_train.csv")
df_component_test.to_csv(f"{save_path}/component_test.csv")
df_component_val.to_csv(f"{save_path}/component_val.csv")
df_relation_train.to_csv(f"{save_path}/relation_train.csv")
df_relation_test.to_csv(f"{save_path}/relation_test.csv")
df_relation_val.to_csv(f"{save_path}/relation_val.csv")
