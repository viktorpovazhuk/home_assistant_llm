import pandas as pd

df = pd.read_csv("functions.csv")

for idx, row in df.iterrows():
    with open(f'docs/{row["name"]}.md', "w") as f:
        f.write(f'Name: {row["name"]}\n')
        f.write(f'Description: {row["description"]}')

df = pd.read_csv("component.csv")

for idx, row in df.iterrows():
    with open(f'docs/{row["name"]}.md', "w") as f:
        f.write(f'Name: {row["name"]}\n')
        f.write(f'Type: {row["type"]}\n')
        f.write(f'Description: {row["description"]}\n')
