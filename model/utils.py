from packages import *

################ model utils ################

class Logger:
    def __init__(self, model, optimizer, root, run_name='default_run'):
        self.model, self.optimizer= model, optimizer
        self.run_name, self.root= run_name, root
        self.history= { 'train_loss': [],
                        'valid_loss': [],
                        'best_loss_valid': float('inf'),
                        'seen_tokens': [] }

    def log(self, train_loss, valid_loss, seen_tokens):
        self.history['train_loss'].append(train_loss)
        self.history['valid_loss'].append(valid_loss)
        self.history['seen_tokens'].append(seen_tokens)

    def save(self):
        file_path= f'{self.root}model/logs/{self.run_name}.json'
        with open(file_path, 'w') as f:
            json.dump(self.history, f, indent= 4)

        current_loss_valid= self.history['valid_loss'][-1]
        if current_loss_valid < self.history['best_loss_valid']:
            log= dict(model= self.model.state_dict(), optimizer= self.optimizer)
            torch.save(log, f'{self.root}model/logs/best-model.pt')
            self.history['best_loss_valid']= current_loss_valid
            print("✅ Model Saved!")

    def plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['seen_tokens'], self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['seen_tokens'], self.history['valid_loss'], label='Valid Loss')
        plt.xlabel('Seen Tokens')
        plt.ylabel('Loss')
        plt.title(f'Training Curve: {self.run_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.root}model/logs/{self.run_name}_curve.png')



def print_config_summary(model, optimizer, loss_fn, train_loader, device, total_tokens, log_interval_tokens):
    table= Table(title="Training Configuration Summary")
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Details", style="magenta")

    model_type= str(model.config).replace("namespace", model.__class__.__name__)
    table.add_row("Model Type", model_type)
    table.add_section()

    total_params= sum(p.numel() for p in model.parameters())
    te_params= model.token_embedding.weight.numel()
    param_summary= f"{total_params:,} ({total_params - te_params:,} + {te_params:,})"
    table.add_row("Total Parameters (Tr+TE)", param_summary)
    table.add_section()

    optimizer_name= optimizer.__class__.__name__
    optimizer_params= ', '.join([f"{k}={v}" for k, v in optimizer.defaults.items() if k in ["lr", "betas", "weight_decay", "fused"]])
    table.add_row("Optimizer", f"{optimizer_name}({optimizer_params})")
    table.add_section()

    loss_name= loss_fn.__name__ if hasattr(loss_fn, '__name__') else str(loss_fn)
    table.add_row("Loss Function", loss_name)
    table.add_section()

    batch_shape= f"{train_loader.batch_size}x{train_loader.dataset[0][0].shape[-1]}"
    table.add_row("Batch Shape", batch_shape)
    table.add_section()

    table.add_row("Device", str(device))
    table.add_row("Max Tokens", f"{total_tokens:,}")
    table.add_row("Log Interval Tokens", f"{log_interval_tokens:,}")
    table.add_section()

    console= Console()
    console.print(table)
    print()



def text_generation_fn(model, tokenizer, prompt, max_length= 128, temperature= 1.5, top_k= 5):
    input_ids= torch.tensor(tokenizer.encode(prompt).ids, dtype= torch.int).unsqueeze(0).to(model.device)
    eos= torch.tensor(tokenizer.encode("<|endoftext|>").ids, dtype= torch.int)[0]
    model.eval()

    with torch.inference_mode():
        for _ in range(max_length):
            logits= model(input_ids)
            scores= (logits[0, [-1]] / temperature).softmax(dim= -1)
            topk_probs, topk_indices= torch.topk(scores, k= top_k, dim= -1)
            ids= torch.multinomial(topk_probs, 1)
            ids= torch.gather(topk_indices, -1, ids)
            input_ids= torch.cat((input_ids, ids), dim= -1)
            if ids.item() == eos.item(): 
                print(" *eos")
                break
            if input_ids.shape[1] == max_length: input_ids= input_ids[:, 1:]

    return tokenizer.decode(input_ids[0].tolist())



def lr_grid_search(model, train_loader, valid_loader, optimizer, loss_fn, device,
                   num_epochs= 1, lr_range= (1e-5, 1e-1), num_lr= 5):
    # Implement learning rate finder logic here
    pass


################ main utils ################

def print_title(title: str):
    s= (70-len(title)-1)//2
    e= 70-len(title)-s-2
    print()
    print(70*"#")
    print(s*"#", title, e*"#")
    print(70*"#")


def num_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)/ 1e6


def read_config(config_name: str, root: str) -> SimpleNamespace:
    with open(f"{root}config/{config_name}.json") as f:
        return SimpleNamespace(**json.load(f))