from packages import *

def one_epoch_fn(data_loader, model, loss_fn, device, status= "train", optimizer= None):
    if status not in ("train", "eval"): raise ValueError("Invalid status, must be 'train' or 'eval'.")
    grad_context= contextlib.nullcontext() if status == "train" else torch.no_grad()
    model.train() if status == "train" else model.eval()

    loss_history= 0
    with grad_context:
        batch_number= 0 #########
        for input_ids, labels in data_loader:
            print(f"Processing batch {batch_number + 1}/{len(data_loader)}") #########
            batch_number+=1 #########
            if batch_number > 3: break #########

            input_ids, labels= input_ids.to(device), labels.to(device)
            logits= model(input_ids)
            loss= loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            loss_history+= loss.item()
            print(loss) #########

            if status== "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return model, loss_history/len(data_loader)



def lr_grid_search(model, train_loader, valid_loader, optimizer, loss_fn, device,
                   num_epochs= 1, lr_range= (1e-5, 1e-1), num_lr= 5):
    # Implement learning rate finder logic here
    pass



def training_fn(model, train_loader, valid_loader, num_epochs, optimizer, loss_fn, device, basic_loss= float("inf")):
    train_loss_history, valid_loss_history= [], []
    model= model.to(device)
    for epoch in range(num_epochs):
        print(10*"- ", f"Epoch {epoch + 1}/{num_epochs}", 10*"- ",)
        model, train_loss= one_epoch_fn(train_loader, model, loss_fn, device, status= "train", optimizer= optimizer)
        print(f"Train loss: {train_loss:.4f}")

        _, valid_loss= one_epoch_fn(valid_loader, model, loss_fn, device, status= "eval")
        print(f"Valid loss: {valid_loss:.4f}")

        if valid_loss < basic_loss:
            print("Validation loss improved, saving model...")
            torch.save(model, "./model/gpt2_model.pt")
            basic_loss= valid_loss

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

    return model, train_loss_history, valid_loss_history



def text_generation_fn(model, tokenizer, prompt, max_length= 128, temperature= 1.0):
    input_ids= torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(model.device)
    model.eval()
    with torch.inference_mode():
        for _ in range(max_length):
            logits= model(input_ids)[:, -1, :] / temperature
            logits= logits.argmax(dim=-1).unsqueeze(-1)
            input_ids= torch.cat([input_ids, logits], dim= -1)
            if logits.item() == tokenizer.eos_token_id: break

    return tokenizer.decode(input_ids.squeeze(0).tolist())



def print_title(title: str):
    s= (70-len(title)-1)//2
    e= 70-len(title)-s-2
    print()
    print(70*"#")
    print(s*"#", title, e*"#")
    print(70*"#")



def num_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)/ 1e6