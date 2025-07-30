from packages import *
from preprocessing import *

os.system('cls' if os.name == 'nt' else 'clear')

# preprocessing
dataset= dataset_loader_fn()

if os.path.exists("custom_tokenizer_1K.json"):
    tokenizer= Tokenizer.from_file("custom_tokenizer_1K.json")
else: tokenizer= tokenizer_fn(dataset)


train_loader= data_preparation(dataset["train"], tokenizer)
valid_loader= data_preparation(dataset["validation"],tokenizer, shuffle_st= False)

torch.save(train_loader, "./preprocessing/train_loader.pt")
torch.save(valid_loader, "./preprocessing/valid_loader.pt")
print(50*"- ", "dataloader", 50*" -")
print("train batch size:",train_loader.batch_size, ", num of batch:", len(train_loader))
print("valid batch size:",valid_loader.batch_size, ", num of batch:", len(valid_loader))