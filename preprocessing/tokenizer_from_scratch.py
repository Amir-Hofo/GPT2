from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers import processors, decoders
from datasets import load_dataset

repo_id= "roneneldan/TinyStories"
dataset= load_dataset(repo_id)

unk, eos= "[UNK]", "<|endoftext|>"
tokenizer= Tokenizer(models.BPE(unk_token= unk))
tokenizer.pre_tokenizer= pre_tokenizers.ByteLevel()
trainer= trainers.BpeTrainer(vocab_size= 10000, min_frequency= 2,
                             special_tokens= [unk, eos] )

tokenizer.train_from_iterator(dataset["train"]["text"], trainer)
print(10 * "--", " vocab size ", 10 * "--")
print(tokenizer.get_vocab_size())

tokenizer.post_processor= processors.TemplateProcessing(
    single= f"{eos} $A {eos}",
    special_tokens= [(eos, tokenizer.token_to_id(eos))]
    )

tokenizer.decoder= decoders.ByteLevel()

tokenizer.save("custom_tokenizer.json")
tokenizer= Tokenizer.from_file("custom_tokenizer.json")

# vocab
vocab= tokenizer.get_vocab()
sorted_vocab= dict(sorted(vocab.items(), key= lambda item: item[1]))
print(10 * "--", " vocab ", 10 * "--", "\n",
*(f"{token}: {index}" for i, (token, index) in enumerate(sorted_vocab.items()) if i< 10),
"...", *(f"{token}: {index}" for token, index in list(sorted_vocab.items())[-5:]), sep=", ")

# test
text= "Mr. Smith bought cheapsite.com for 12.5$ milion in 2023."
encoded_text= tokenizer.encode(text)
print(10 * "--", " test ", 10 * "--")
print("text: ", text)
print("tokens: ", encoded_text.tokens)
print("ids: ", encoded_text.ids)
print("decoded: ", tokenizer.decode(encoded_text.ids))