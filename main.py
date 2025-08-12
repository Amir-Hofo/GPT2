from packages import *
from preprocessing import *
from model import *

os.system('cls' if os.name == 'nt' else 'clear')
device= "cuda" if torch.cuda.is_available() else "cpu"
project_root= "./"

############## Configuration ##############
preprocess_config= read_config("preprocess_config", project_root)
model_config= read_config("model_config", project_root)
train_config= read_config("train_config", project_root)

############## Preprocessing ##############
print_title("Preprocessing")

dataset= dataset_loader_fn(preprocess_config)

if os.path.exists(f"{project_root}preprocessing/custom_tokenizer_10K.json"):
    tokenizer= Tokenizer.from_file(f"{project_root}preprocessing/custom_tokenizer_10K.json")
else: tokenizer= tokenizer_fn(dataset, preprocess_config, project_root)

if os.path.exists(f"{project_root}preprocessing/train_loader.pt"):
    train_loader= torch.load(f"{project_root}preprocessing/train_loader.pt")
else: 
    train_loader= data_preparation(dataset["train"], tokenizer, preprocess_config)
    torch.save(train_loader, f"{project_root}preprocessing/train_loader.pt")

if os.path.exists(f"{project_root}preprocessing/valid_loader.pt"):
    valid_loader= torch.load(f"{project_root}preprocessing/valid_loader.pt")
else:
    valid_loader= data_preparation(dataset["validation"],tokenizer, preprocess_config, shuffle_st= False)
    torch.save(valid_loader, f"{project_root}preprocessing/valid_loader.pt")

print(10*"- ", "dataloader", 10*" -")
print("train batch size:",train_loader.batch_size, ", num of batch:", len(train_loader))
print("valid batch size:",valid_loader.batch_size, ", num of batch:", len(valid_loader))

x, y= next(iter(train_loader))
print("sample data:", x.shape, y.shape)


################## Model ##################
print_title("Model")


model= GPT2Model(model_config)
model.device= device
print("Number of model parameters: ", num_trainable_params(model),"M")
print(model)


################# Training ################
print_title("Training")

loss_fn= nn.CrossEntropyLoss()
optimizer= torch.optim.AdamW(model.parameters(), lr= train_config.learning_rate)

model_trainer= ModelTrainer(model, train_loader, valid_loader, optimizer, loss_fn, train_config, project_root)
model_trainer.training()


############## Text Generation ############
prompt= "i am in danger"
for _ in range(3):
    text= model.text_generator(prompt, tokenizer,  max_length= 10, temperature= 1.5)
    print(colored(prompt, "cyan"), text)