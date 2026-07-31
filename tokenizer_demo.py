from tokenizer import Tokenizer
import tiktoken

my_tokenizer = Tokenizer()
tiktoken_tokenizer = tiktoken.get_encoding("gpt2")

sentence = "This is \n some text"

tiktoken_tokens = tiktoken_tokenizer.encode(sentence)
my_tokens = my_tokenizer.encode(sentence)

my_text = my_tokenizer.decode(my_tokens)
tiktoken_text = tiktoken_tokenizer.decode(tiktoken_tokens)

assert sentence == my_text == tiktoken_text, "Decoded texts do not match."
assert my_tokens == tiktoken_tokens, "My tokens do not match the TikToken tokens."

print("Sentence:", sentence)
print("Tokens (My Tokenizer):", my_tokens)
print("Decoded (My Tokenizer):", my_text)