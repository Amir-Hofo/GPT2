from huggingface_hub import hf_hub_download
import os, tarfile, json
import pandas as pd

repo_id= "roneneldan/TinyStories"
local_dir= "./dataset/"
file_name= "TinyStories_all_data.tar.gz"
extract_dir= "./dataset/TinyStories_all_data"

zip_file= hf_hub_download(repo_id= repo_id, repo_type= "dataset",
                          filename= file_name, local_dir= local_dir)
print("the file was downloaded to {zip_file}")

os.makedirs(extract_dir, exist_ok= True)
with tarfile.open(local_dir+ file_name, "r:gz") as tar:
    tar.extractall(path= extract_dir)
print("extraction completed.")

with open('./dataset/TinyStories_all_data/data00.json', 
          'r', encoding= 'utf-8') as f:
    data= json.load(f)

df= pd.DataFrame(data)
df= pd.concat([df.drop(columns= ['instruction']), 
               df['instruction'].apply(pd.Series)], axis= 1)
print(df.head())

df.to_excel('./dataset/data00.xlsx', index= False)