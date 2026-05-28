# -*- coding: utf-8 -*-
import os
import re
import logging
from sklearn.metrics import f1_score, cohen_kappa_score, mean_squared_error
import pandas as pd
logging.basicConfig(level=logging.INFO)
import pasp

# Silence other loggers
for log_name, log_obj in logging.Logger.manager.loggerDict.items():
    log_obj.disabled = True

logging.info("Training Argument Quality Task")

file_path = "./dataset/"

# Argument Quality
df_quality_test = pd.read_csv(file_path+"arg_quality_test.csv", header=0, index_col=0)[["text_id", "text_tokens", "arg_score", "arg_quality_fine_grained", "arg_quality"]]
df_quality_test = df_quality_test[df_quality_test.text_id.str.contains("AAEC")] # can be ArGPT
test_ids = df_quality_test.text_id.unique()

df_component_test = pd.read_csv(file_path+"component_test.csv", header=0, index_col=0)[["text_id", "component_tokens", "text_tokens", "minimalist_labels"]]
df_component_test.rename(columns={"minimalist_labels": "labels"}, inplace=True)
df_component_test["tokens"]= df_component_test.component_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())

df_relation_test = pd.read_csv(file_path+"relation_test.csv", header=0, index_col=0)[["text_id", "source_tokens", "target_tokens", "minimalist_labels"]]
df_relation_test.rename(columns={"minimalist_labels": "labels"}, inplace=True)
df_relation_test["source_tokens"]= df_relation_test.source_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())
df_relation_test["target_tokens"]= df_relation_test.target_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())
relations_ids = df_relation_test.text_id.unique()

# Quality of Relations
df_component_test_2 = pd.read_csv(file_path+"component_rel_quality_test.csv", header=0, index_col=0)[["text_id", "component_tokens", "supports_strenght", "attacks_strenght"]]
df_component_test_2["tokens"]= df_component_test_2.component_tokens.apply(lambda x:re.sub(r'[^\w]', ' ', x).strip().replace(" ", "").lower())
df_component_test_2 = df_component_test_2[["text_id", "tokens", "supports_strenght", "attacks_strenght"]]

df_component_test = pd.merge(df_component_test, df_component_test_2,  how='left', left_on=['text_id', "tokens"], right_on = ['text_id', "tokens"])

logging.info("Processing argumentative structures \n")

# Depending on the num of labels for the SAQ Module
#value_set = [0, 0.4, 0.6, 0.8, 1.0]
#value_set = [0, 0.5, 1.0]
value_set = [0, 1.0]
#value_set = [1.0]

def return_closest(n, value_set=value_set):
    return min(value_set, key=lambda x:abs(x-n))

predicted_labels = list()
true_2 = list()
pred_2 = list()
true_4 = list()
pred_4 = list()
true_70 = list()
pred_70 = list()
true_reg = list()
pred_reg = list()

for i, text_id in enumerate(test_ids):

    # Get true labels
    df_true = df_quality_test[df_quality_test.text_id == text_id]
    true_2.append(df_true.arg_quality.to_list()[0].strip())
    true_4.append(df_true.arg_quality_fine_grained.to_list()[0].strip())
    score = df_true.arg_score.to_list()[0]
    true_reg.append(score)
    logging.info(score)
    if int(score) >= 70:
        true_70.append("good")
    else:
        true_70.append("bad")
    
    # Preprocess text_id
    if len(text_id.replace("AAEC_", "")) == 1:
        text_id = "AAEC_00" + text_id.replace("AAEC_", "")
    elif len(text_id.replace("AAEC_", "")) == 2:
        text_id = "AAEC_0" + text_id.replace("AAEC_", "")

    logging.info(f"Evaluation {i+1} out of {len(test_ids)} --> {text_id}\n")
    df_comp = df_component_test[df_component_test.text_id == text_id]
    components = df_comp.to_dict('records')
    df_rel = df_relation_test[df_relation_test.text_id == text_id]
    relations = df_rel.to_dict('records')
    df_rel = df_relation_test[df_relation_test.text_id == text_id]
    df_supports = df_rel[df_rel.labels == "support"]
    supports = df_supports.to_dict('records')
    df_attack = df_rel[df_rel.labels == "attack"]
    attacks = df_attack.to_dict('records')

    # Detecting Major Claims and creating atoms from arguments
    m_claims = list()
    comp_to_atoms = dict()

    for num, component in enumerate(components):
        atom = f"a{num}"
        key  = component["tokens"]
        label = component["labels"]
        if label.strip() == "MajorClaim":
            m_claims.append(atom)
            #logging.info(atom)

        comp_to_atoms[key] = atom
    if len(m_claims) == 0:
        # If there are no MajorClaims, the text is bad at argumentation
        logging.info(f"{text_id} --> No MajorClaim \n")
        pred_2.append("bad")
        pred_4.append("bad")
        pred_70.append("bad")
        pred_reg.append(0)
        continue
    
    logging.info(f"{text_id} --> Creating logic program equivalent to argumentation graph \n")
    
    program = list()

    for component in components:
        component_label = component["labels"]
        key  = component["tokens"]
        # Depends on whether SAQ support and attack are activated or not
        #supports_quality = 1.0
        supports_quality = return_closest(component["supports_strenght"], value_set=value_set)
        #supports_quality = component["supports_strenght"]
        #attacks_quality = return_closest(component["attacks_strenght"], value_set=value_set)
        #attacks_quality = component["attacks_strenght"]
        attacks_quality = 1.0

        target_atom = comp_to_atoms[key]
        program_line = f"supports({target_atom}) :- supports_quality({target_atom}),"
        has_support = False
        has_attack = False

        for relation in supports:
            if relation["target_tokens"] == key:
                source_key = relation["source_tokens"]
                label = relation["labels"]
                source_atom = comp_to_atoms[source_key]
            else:
                continue
            if target_atom in m_claims and source_atom in m_claims:
                continue
            if target_atom == source_atom:
                continue
            program_line = f"{program_line} acceptable({source_atom}),"
            has_support = True
        if has_support:
            program_line = program_line[:-1]
            program_line = program_line + "."
            program.append(program_line)
            if float(supports_quality) == 0.0:
                program.append(f":- supports_quality({target_atom}).")
            elif float(supports_quality) < 1.0:
                program.append(f"{supports_quality}:: supports_quality({target_atom}).")
            else:
                program.append(f"supports_quality({target_atom}).")
        
        program_line = f"failed_attacks({target_atom}) :-"
        for relation in attacks:
            if relation["target_tokens"] == key:
                source_key = relation["source_tokens"]
                label = relation["labels"]
                source_atom = comp_to_atoms[source_key]
            else:
                continue
            program_line = f"{program_line} not acceptable({source_atom}),"
            has_attack = True
        if has_attack:
            program_line = program_line[:-1]
            program_line = program_line + "."
            program.append(program_line)
            if float(attacks_quality) == 0.0:
                program.append(f":- attacks_quality({target_atom}).")
            elif float(attacks_quality) < 1.0:
                program.append(f"{attacks_quality}:: attacks_quality({target_atom}).") 
            else:
                program.append(f"attacks_quality({target_atom}).")
            program.append(f"attacks({target_atom}) :- attacks_quality({target_atom}), not failed_attacks({target_atom}).")

        if has_attack and has_support:
            program.append(f"acceptable({target_atom}) :- supports({target_atom}), not attacks({target_atom}).")
        elif has_attack and not has_support:
            program.append(f"acceptable({target_atom}) :- not attacks({target_atom}).")
        elif not has_attack and has_support:
            program.append(f"acceptable({target_atom}) :- supports({target_atom}).")
        else:
            program.append(f"acceptable({target_atom}).")
    
    program.append("#semantics lstable.")

    for claim in m_claims:
        program.append(f"#query(acceptable({claim})).")
    program_str = "\n".join(program)

    logging.info(program)

    logging.info(f"{text_id} --> Executing Reasoning \n")
    P = pasp.parse(program_str, from_str=True)
    results = P(quiet=True, status=False)

    logging.info(f"{text_id} --> Creating quality logic program \n")
    new_program = list()
    last_line = "consistent :-"

    for i, claim in enumerate(m_claims):
        new_program.append(f"majorclaim({claim}).")
        new_program.append(f"[{results[i][0]}, {results[i][1]}] :: accepted({claim}).")
        new_program.append(f"consistent({claim}) :- majorclaim({claim}), accepted({claim}).")
        last_line = last_line + f" consistent({claim}),"

    last_line = last_line[:-1]
    last_line = last_line + "."
    new_program.append(last_line)
    new_program.append("#semantics lstable.")
    new_program.append("#query(consistent).")
    new_program_str = "\n".join(new_program)

    logging.info(f"{text_id} --> Executing Quality Program \n")
    P_2 = pasp.parse(new_program_str, from_str=True)
    final_results = P_2(quiet=True, status=False)
    
    logging.info(f"{text_id} --> Assessing Text Argumentation Quality \n")
    lower_bound = float(final_results[0][0])
    higher_bound = float(results[0][1])
    arg_score = 100*(lower_bound+higher_bound)/2
    
    if arg_score > 75:
        arg_quality_fine_grained = "good"
        arg_quality = "good"
    elif arg_score > 50:
        arg_quality_fine_grained = "mostly good"
        arg_quality = "good"
    elif arg_score > 25:
        arg_quality_fine_grained = "mostly bad"
        arg_quality = "bad"
    else:
        arg_quality_fine_grained = "bad"
        arg_quality = "bad"
    
    pred_reg.append(arg_score)
    pred_2.append(arg_quality)
    pred_4.append(arg_quality_fine_grained)

    if int(arg_score) >= 70:
        pred_70.append("good")
    else:
        pred_70.append("bad")


    logging.info(f"{text_id} --> {arg_quality_fine_grained} and {arg_quality}")

logging.info("\n Final Results - 2 labels 50%\n")

f1_macro = f1_score(average="macro", y_true=true_2, y_pred=pred_2)
f1 = f1_score(average="micro", y_true=true_2, y_pred=pred_2)

logging.info(f"F1 : {f1}  ")
logging.info(f"F1 Macro : {f1_macro}")
      
logging.info("\n Final Results - 4 labels 50%\n")

f1_macro = f1_score(average="macro", y_true=true_4, y_pred=pred_4)
f1 = f1_score(average="micro", y_true=true_4, y_pred=pred_4)

logging.info(f"F1 : {f1}  ")
logging.info(f"F1 Macro : {f1_macro}")

logging.info("\n Final Results - 2 labels 70%\n")

f1_macro = f1_score(average="macro", y_true=true_70, y_pred=pred_70)
f1 = f1_score(average="micro", y_true=true_70, y_pred=pred_70)

logging.info(f"F1 : {f1}  ")
logging.info(f"F1 Macro : {f1_macro}")

logging.info("\n Final Results - Regression\n")

mse = mean_squared_error(true_reg, pred_reg)
qwk_labels = [int(i) * 100 for i in true_reg]
qwk_logits = [int(i) * 100 for i in pred_reg]
qwk = cohen_kappa_score(qwk_labels, qwk_logits, weights="quadratic")

logging.info(f"MSE : {mse}")
logging.info(f"QWK : {qwk}")
