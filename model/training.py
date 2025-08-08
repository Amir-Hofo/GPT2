from packages import *
from model.utils import Logger
from model.utils import print_config_summary

class ModelTrainer():
    def __init__(self, model, train_loader, valid_loader, optimizer, loss_fn, config, project_root):

        self.model, self.train_loader, self.valid_loader, self.device= model, train_loader, valid_loader, model.device
        self.optimizer, self.loss_fn, self.project_root= optimizer, loss_fn, project_root
        self.total_tokens, self.log_interval_tokens= int(config.total_tokens), int(config.log_interval_tokens)
        self.seen_tokens, self.token_eval_counter, self.basic_loss= 0, 0, float("inf")
        self.logger= Logger(self.model, self.optimizer, project_root, 'gpt2_tinystories')
        print_config_summary(self.model, self.optimizer, self.loss_fn, self.train_loader, 
                             self.device, self.total_tokens, self.log_interval_tokens)
    

    def training(self):
        loss_train= MeanMetric()
        self.model.train()
        train_iter= cycle(self.train_loader)
        self.basic_loss= self.evaluate()

        with tqdm(total= self.total_tokens, desc= "Training", unit= "tokens") as pbar:
            while self.seen_tokens < self.total_tokens:
                inputs, targets= next(train_iter)
                inputs, targets= inputs.to(self.device), targets.to(self.device)
                logits= self.model(inputs)
                loss= self.loss_fn(logits.view(-1, logits.shape[-1]), targets.flatten())
                
                loss.backward()
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), max_norm= 1.)
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_train.update(loss.item(), inputs.shape[0])

                num_tokens_this_batch= inputs.numel()
                self.seen_tokens+= num_tokens_this_batch
                self.token_eval_counter+= num_tokens_this_batch

                pbar.set_postfix({"Loss": f"{loss_train.compute().item():.4f}",
                                  "LR"  : f"{self.optimizer.param_groups[0]['lr']:.2e}",})
                pbar.update(num_tokens_this_batch)

                if (self.token_eval_counter >= self.log_interval_tokens) or (self.seen_tokens >= self.total_tokens):
                    loss_valid, self.token_eval_counte= self.evaluate(), 0
                    print(f"\nValid Loss: {loss_valid:.4f}")

                    self.logger.log(loss_train.compute().item(), loss_valid, self.seen_tokens)
                    self.logger.save()
                    if loss_valid < self.basic_loss:
                        torch.save(self.model, f"{self.project_root}model/gpt2_model.pt")
                        self.basic_loss= loss_valid
                        print("Validation loss improved, saving model...")
        
        self.logger.plot()
                        

    def evaluate(self):
        loss_valid= MeanMetric()
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in self.valid_loader:
                inputs, targets= inputs.to(self.device), targets.to(self.device)
                logits= self.model(inputs)
                loss= self.loss_fn(logits.view(-1, logits.shape[-1]), targets.flatten())
                loss_valid.update(loss.item(), inputs.shape[0])
                break
        return loss_valid.compute().item()
