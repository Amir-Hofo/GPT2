from packages import *
from preprocessing import *
from model import *

os.system('cls' if os.name == 'nt' else 'clear')
device= "cuda" if torch.cuda.is_available() else "cpu"

############## Preprocessing ##############
print_title("Preprocessing")
dataset= dataset_loader_fn()

if os.path.exists("./preprocessing/custom_tokenizer_10K.json"):
    tokenizer= Tokenizer.from_file("./preprocessing/custom_tokenizer_10K.json")
else: tokenizer= tokenizer_fn(dataset)

if os.path.exists("./preprocessing/train_loader.pt"):
    train_loader= torch.load("./preprocessing/train_loader.pt")
else: 
    train_loader= data_preparation(dataset["train"], tokenizer)
    torch.save(train_loader, "./preprocessing/train_loader.pt")

if os.path.exists("./preprocessing/valid_loader.pt"):
    valid_loader= torch.load("./preprocessing/valid_loader.pt")

else:
    valid_loader= data_preparation(dataset["validation"],tokenizer, shuffle_st= False)
    torch.save(valid_loader, "./preprocessing/valid_loader.pt")

print(10*"- ", "dataloader", 10*" -")
print("train batch size:",train_loader.batch_size, ", num of batch:", len(train_loader))
print("valid batch size:",valid_loader.batch_size, ", num of batch:", len(valid_loader))


################## Model ##################
print_title("Model")

config= {"num heads": 2, "feature dimension": 10, 
         "mha dropout": 0.1, "ffnn dropout": 0.1,
         "vocab size": 50, "max position": 25, #"max position": seq_len
         "num layers": 2, "dropout": 0.1}
model= GPT2Model(config)
print(num_trainable_params(model))
input= torch.randint(0, config["vocab size"], (4, 3), dtype= torch.long)
print(model(input).shape)