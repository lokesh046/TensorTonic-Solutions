import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        self.word_to_id = {

            "<PAD>": 0,

            "<UNK>": 1,

            "<BOS>": 2,

            "<EOS>": 3

        }

        self.id_to_word = {

            0: "<PAD>",

            1: "<UNK>",

            2: "<BOS>",

            3: "<EOS>"

        }
        
        unique_words = set()

        for sentence  in texts:
            words = sentence.lower().split()
            unique_words.update(words)
        sorted_words = sorted(unique_words)

        for idx,word in enumerate(sorted_words,start=4):
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word

        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        words =  text.lower().split()

        ids = []

        for word in words:
            ids.append(self.word_to_id.get(word,1))

        return ids
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        words = []

        for token_ids in ids:
            words.append(self.id_to_word.get(token_ids,"<UNK>"))

        return " ".join(words)