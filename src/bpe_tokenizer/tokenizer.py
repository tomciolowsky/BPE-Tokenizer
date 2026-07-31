import json
import pathlib
import re

DATA_DIR = pathlib.Path(__file__).parent / "data"

class Tokenizer:
    def __init__(self, data_dir:pathlib.Path=DATA_DIR):
        self.data_dir = data_dir
        self.token_to_id = {}
        self.id_to_token = {}
        self.ranks = {}
        self.load_dictionary()
        self.load_ranks()

    def load_dictionary(self, file_name="encoder.json"):
        """
        Load the vocabulary from a JSON file and create mappings between tokens and IDs.
        """
        dictionary_path = self.data_dir / file_name
        with open(dictionary_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
            self.id_to_token = {int(v): k for k, v in vocab.items()}
            self.token_to_id = {k: int(v) for k, v in vocab.items()}

            self.token_to_id["\n"] = 198
            self.token_to_id["\n\n"] = 628
            self.token_to_id["\r"] = 201

    def load_ranks(self, file_name="vocab.bpe"):
        """
        Load the BPE ranks from a file and create a mapping between token pairs and their ranks.
        """
        ranks_path = self.data_dir / file_name
        with open(ranks_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

            if lines[0][0] == "#":
                lines = lines[1:]

            rank = 0
            for line in lines:
                token1, token2 = line.strip().split()
                if token1 in self.token_to_id and token2 in self.token_to_id:
                    self.ranks[(token1, token2)] = rank
                    rank += 1

    def text_to_tokens(self, text:str) -> list[str]:
        """
        Convert given text into a list of tokens
        """
        tokens = []
        parts = re.split(r'(\r\n|\n\n|\r|\n)', text)
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
            if part == "\n\n":
                tokens.append("\n\n")
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
        """
        Tokenize a single token using BPE
        """
        token_ids = [self.token_to_id.get(character, None) for character in token]
        if None in token_ids:
            raise ValueError(f"Token '{token}' contains characters not found in vocabulary.")

        strings = [self.id_to_token[token_id] for token_id in token_ids]

        while True:
            if len(strings) == 1:
                break

            pairs_strings = set([(strings[i], strings[i + 1]) for i in range(len(strings) - 1)])
            if not pairs_strings:
                break

            pairs_ranks = [(pair, self.ranks.get(pair, float('inf'))) for pair in pairs_strings]
            best_pair, best_rank = min(pairs_ranks, key=lambda x: x[1])
            if best_rank == float('inf'):
                break

            pair_1, pair_2 = best_pair
            new_strings = []

            j = 0
            while j < len(strings):
                if j < len(strings) - 1 and strings[j] == pair_1 and strings[j + 1] == pair_2:
                    new_strings.append(pair_1 + pair_2)
                    j += 2
                else:
                    new_strings.append(strings[j])
                    j += 1

            strings = new_strings

        new_token_ids = [self.token_to_id[string] for string in strings]
        return new_token_ids

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
            elif token_id == 628:
                tokens.append("\n\n")
                
            elif token[0] == "Ġ":
                tokens.append(" " + token[1:])
            else:
                tokens.append(token)
            
        return "".join(tokens)