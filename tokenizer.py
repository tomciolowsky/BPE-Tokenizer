import json
import re

class Tokenizer:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.load_dictionary()

    def load_dictionary(self, file_name="encoder.json"):
        """
        Load the vocabulary from a JSON file and create mappings between tokens and IDs.
        """
        with open(file_name, "r", encoding="utf-8") as f:
            vocab = json.load(f)
            self.id_to_token = {int(v): k for k, v in vocab.items()}
            self.token_to_id = {k: int(v) for k, v in vocab.items()}

            self.token_to_id["\n"] = 198
            self.token_to_id["\r"] = 201

    def text_to_tokens(self, text:str) -> list[str]:
        """
        Convert given text into a list of tokens
        """
        tokens = []
        parts = re.split(r'(\r\n|\r|\n)', text)
        for part in parts:
            if part == "":
                continue
            if part == "\r\n":
                tokens.append("\r")
                tokens.append("\n")
                continue
            if part == "\r":
                tokens.append("\r")
                continue
            if part == "\n":
                tokens.append("\n")
                continue

            s = re.split(r'(\S+)', part)
            
            if s[0] == "":
                s = s[1:]
            if s[-1] == "":
                s = s[:-1]

            for i, string in enumerate(s):

                if string[0] == " ":
                    k = len(string)-1 if i < len(s)-1 else len(string)
                    tokens += ["Ġ"] * k

                elif i == 0:
                    tokens.append(string)
                    
                else:
                    tokens.append("Ġ"+string)
            
        return tokens

    def bpe_tokenize(self, token:str) -> list[int]:
        raise NotImplementedError("BPE tokenization is not implemented yet.")

    def encode(self, text:str) -> list[int]:
        """
        Encode given text into a list of token IDs
        """
        token_ids = []
        tokens = []

        tokens = self.text_to_tokens(text)

        for token in tokens:
            if token not in self.token_to_id:
                token_ids += self.bpe_tokenize(token)
            else:
                token_ids.append(self.token_to_id[token])

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode given list of token IDs into text
        """
        tokens = []
        for token_id in token_ids:
            if token_id not in self.id_to_token:
                raise ValueError(f"Token ID '{token_id}' not found in vocabulary.")

            token = self.id_to_token[token_id]

            if token_id == 198:
                tokens.append("\n")
            elif token_id == 201:
                tokens.append("\r")
                
            elif token[0] == "Ġ":
                tokens.append(" " + token[1:])
            else:
                tokens.append(token)
            
        return "".join(tokens)