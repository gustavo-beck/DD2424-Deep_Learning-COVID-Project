import numpy as np
import pandas as pd
from tqdm import tqdm

PATH = "~/Git_Projects/DD2424-Deep_Learning-COVID-Project/covid_dataset/"
df = pd.read_csv(PATH + "metadata.csv")
df = df[(df['modality'] == "X-ray") & (df['view'] != "L")]
df = df.reset_index()
df = df[['finding', 'filename']]

findings_dict = {}
for i, sample in tqdm(df.iterrows()):
    findings = sample['finding'].split(', ')
    for finding in findings:
        if finding not in findings_dict.keys():
            findings_dict[finding] = len(findings_dict.keys())
print(findings_dict)

cols = list(findings_dict.keys())

# cols.extend(findings_dict.keys())

new_df = pd.DataFrame(df['filename'])
for col in cols:
    new_df[col] = 0


for i, sample in tqdm(df.iterrows()):
    findings = sample['finding'].split(', ')
    #print(new_df['Image Index'].iloc[i], new_df['Image Index'].iloc[i])
    for finding in findings:
        new_df.iloc[i, 1 + findings_dict[finding]] = 1

new_df.to_csv(PATH + "organized_dataset.csv", index=False)

#new_df.to_csv(PATH + "organized_dataset.csv")