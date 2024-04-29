# coding: utf-8

"""
Vocabulary module
"""
from collections import defaultdict, Counter
from typing import List
import numpy as np
import csv
import re

from constants import UNK_TOKEN, DEFAULT_UNK_ID, \
    EOS_TOKEN, BOS_TOKEN, PAD_TOKEN


class Vocabulary:
    """ Vocabulary represents mapping between tokens and indices. """

    def __init__(self, tokens: List[str] = None, file: str = None) -> None:

        # don't rename stoi and itos since needed for torchtext
        # warning: stoi grows with unknown tokens, don't use for saving or size

        # special symbols
        self.specials = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

        self.stoi = defaultdict(DEFAULT_UNK_ID) # string to index
        self.itos = [] # index to string
        if tokens is not None:
            self._from_list(tokens)
        elif file is not None:
            self._from_file(file)

    def _from_list(self, tokens: List[str] = None) -> None:
        """
        Make vocabulary from list of tokens.
        Tokens are assumed to be unique and pre-selected.
        Special symbols are added if not in list.

        :param tokens: list of tokens
        """
        self.add_tokens(tokens=self.specials+tokens)
        assert len(self.stoi) == len(self.itos)

    def _from_file(self, file: str) -> None:
        """
        Make vocabulary from contents of file.
        File format: token with index i is in line i.

        :param file: path to file where the vocabulary is loaded from
        """
        tokens = []
        with open(file, "r") as f:
            tokens = [line.strip() for line in f]
        self.from_list(tokens)

    def __str__(self) -> str:
        return self.stoi.__str__()

    def to_file(self, file: str) -> None:
        """
        Save the vocabulary to a file, by writing token with index i in line i.

        :param file: path to file where the vocabulary is written
        """
        with open(file, "w") as f:
            for token in self.itos:
                f.write(f"{token}\n")

    def add_tokens(self, tokens: List[str]) -> None:
        """
        Add list of tokens to vocabulary
        :param tokens: list of tokens to add to the vocabulary
        """
        for token in tokens:
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def is_unk(self, token: str) -> bool:
        """
        Check whether a token is covered by the vocabulary

        :param token:
        :return: True if covered, False otherwise
        """
        # return self.stoi.get(token, DEFAULT_UNK_ID()) == DEFAULT_UNK_ID()
        return self.stoi[token] == DEFAULT_UNK_ID()

    def __len__(self) -> int:
        return len(self.itos)

    def array_to_sentence(self, array: np.array, cut_at_eos=True) -> List[str]:
        """
        Converts an array of IDs to a sentence. Optional argument cuts out the special token
        at the end of the sentence.

        :param array: 1D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :return: list of strings (tokens)
        """
        sentence = []
        for i in array:
            s = self.itos[i]
            if cut_at_eos and s == EOS_TOKEN:
                break
            sentence.append(s)
        return sentence

    def arrays_to_sentences(self, arrays: np.array, cut_at_eos=True) \
            -> List[List[str]]:
        """
        Convert multiple arrays containing sequences of token IDs to their
        sentences, optionally cutting them off at the end-of-sequence token.

        :param arrays: 2D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :return: list of list of strings (tokens)
        """
        sentences = []
        for array in arrays:
            sentences.append(
                self.array_to_sentence(array=array, cut_at_eos=cut_at_eos))
        return sentences
    
    def tokenize(self, text):
            """ 
            Use regex to separate words from punctuation and tokenize the text. 
            """
            # Pattern to find words or punctuation
            #? Should we include punctuation in the vocabulary?
            text = text.lower() 
            pattern = re.compile(r"[\w']+|[.,!?;]")
            return pattern.findall(text)


    def build_vocab(self, samples: List[str], max_size: int = 10000, min_freq: int = 1):
        """
        Builds vocabulary from a list of sentences. 
        """
        token_counts = Counter(token for sentence in sentences for token in self.tokenize(sentence))
        # Include tokens that meet the minimum frequency
        filtered_tokens = [token for token, count in token_counts.items() if count >= min_freq]
        # Limit to max_size most common tokens
        sorted_tokens = sorted(filtered_tokens, key=lambda token: (-token_counts[token], token))[:max_size]
        self.add_tokens(self.specials+sorted_tokens)

    def get_sentences(self):
        sentences = []
        csv_path = "../data/val/raw/how2sign_realigned_val.csv"
        with open(csv_path, 'r') as file:
            # Create a CSV reader object
            reader = csv.reader(file)

            for row in reader:
                split_row = row[0].split('\t')
                sentence = split_row[-1]
                sentences.append(sentence)

        # print(f"SENTENCES: {sentences}") # Looks all good, some works are stand alone which is odd but should not matter
        return sentences

if __name__ == "__main__":
    vocab = Vocabulary()
    sentences = vocab.get_sentences()
    vocab.build_vocab(sentences)
    # generate the file
    vocab.to_file("../configs/src_vocab.txt")

    print("Index of 'UNK':", vocab.stoi['<unk>'])
    print("Word for index 0:", vocab.itos[0])
    print("Index of 'this':", vocab.stoi['this'])
    print("Word for index 19:", vocab.itos[19])
    print("Vocabulary Size:", len(vocab))
    # print("Vocabulary itos:", vocab.itos) # Works - returns a vocabulary
    # print("Vocabulary stoi:", vocab.stoi) # Works - shows each word mapped to an index