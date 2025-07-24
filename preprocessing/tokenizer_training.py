from packages import *

def tokenizer_fn(dataset, vocab_size= 10_000, min_frequency= 2, save_tokenizer= True):
    unk, eos= "|<unk>|", "<|endoftext|>"
    tokenizer= Tokenizer(models.BPE(unk_token= unk))
    tokenizer.pre_tokenizer= pre_tokenizers.ByteLevel(add_prefix_space= False)
    trainer= trainers.BpeTrainer(vocab_size= vocab_size, 
                                 min_frequency= min_frequency,
                                 special_tokens= [unk, eos] )

    tokenizer.train_from_iterator(dataset["train"]["text"], trainer)
    print(10 * "--", " vocab size ", 10 * "--")
    print(tokenizer.get_vocab_size())

    tokenizer.post_processor= processors.TemplateProcessing(
        single= f"{eos} $A",
        special_tokens= [(eos, tokenizer.token_to_id(eos))]
        )

    tokenizer.decoder= decoders.ByteLevel(add_prefix_space= False)

    if save_tokenizer: tokenizer.save(f"custom_tokenizer_{vocab_size//1000}K.json")
    return tokenizer