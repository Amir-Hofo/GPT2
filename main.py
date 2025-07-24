from packages import *
from dataset import *
from preprocessing import *

os.system('cls' if os.name == 'nt' else 'clear')

dataset= dataset_loader_fn()

if os.path.exists("custom_tokenizer_1K.json"):
    tokenizer= Tokenizer.from_file("custom_tokenizer_1K.json")
else: tokenizer= tokenizer_fn(dataset)


data= CustomDataset(dataset["validation"], tokenizer, 128)
x, y= data[2]
print(x.shape, y.shape)
print(x, y)