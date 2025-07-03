from packages import *

repo_id= "roneneldan/TinyStories"
data= load_dataset(repo_id)
print(data['validation'][12]['text'])
