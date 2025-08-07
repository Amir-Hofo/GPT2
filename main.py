from packages import *
from preprocessing import *
from model import *

os.system('cls' if os.name == 'nt' else 'clear')
device= "cuda" if torch.cuda.is_available() else "cpu"

############## Preprocessing ##############
print_title("Preprocessing")

with open("config/preprocess_config.json") as f: 
    preprocess_config= SimpleNamespace(**json.load(f))

dataset= dataset_loader_fn(preprocess_config)

if os.path.exists("./preprocessing/custom_tokenizer_10K.json"):
    tokenizer= Tokenizer.from_file("./preprocessing/custom_tokenizer_10K.json")
else: tokenizer= tokenizer_fn(dataset, preprocess_config)

if os.path.exists("./preprocessing/train_loader.pt"):
    train_loader= torch.load("./preprocessing/train_loader.pt")
else: 
    train_loader= data_preparation(dataset["train"], tokenizer, preprocess_config)
    torch.save(train_loader, "./preprocessing/train_loader.pt")

if os.path.exists("./preprocessing/valid_loader.pt"):
    valid_loader= torch.load("./preprocessing/valid_loader.pt")
else:
    valid_loader= data_preparation(dataset["validation"],tokenizer, preprocess_config, shuffle_st= False)
    torch.save(valid_loader, "./preprocessing/valid_loader.pt")

print(10*"- ", "dataloader", 10*" -")
print("train batch size:",train_loader.batch_size, ", num of batch:", len(train_loader))
print("valid batch size:",valid_loader.batch_size, ", num of batch:", len(valid_loader))


x, y= next(iter(train_loader))
print("sample data:", x.shape, y.shape)

################## Model ##################
print_title("Model")

with open("config/model_config.json") as f: 
    custom_gpt2_config= SimpleNamespace(**json.load(f))

model= GPT2Model(custom_gpt2_config)
print(num_trainable_params(model))
# print(model)

loss_fn= nn.CrossEntropyLoss()
if custom_gpt2_config.learning_rate == None:
    optimizer= torch.optim.AdamW(model.parameters(), lr= 5e-4) 
    custom_gpt2_config.learning_rate= lr_grid_search(model, train_loader, valid_loader, optimizer, loss_fn, device)
optimizer= torch.optim.AdamW(model.parameters(), lr= custom_gpt2_config.learning_rate)

model, train_loss_history, valid_loss_history= training_fn(model, train_loader, valid_loader, 1, optimizer, loss_fn, device)
print(valid_loss_history[-1])

model.device= device

print(24*" - ")
prompt= ["Once upon a time", "my name is"]
print(text_generation_fn(model, tokenizer, prompt[0]))