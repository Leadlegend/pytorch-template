import os
import collections
from typing import List, Union


def load_vocab(vocab_file, has_index=False):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
        if not has_index:
            for index, token in enumerate(tokens):
                token = token.rstrip("\n")
                vocab[token] = index
        else:
            for i, datas in enumerate(tokens):
                index, token = datas.rstrip('\n').split('\t')
                try:
                    index = int(index)
                except:
                    raise ValueError(
                        'Invalid Vocabulary Item %s at line %d' % (datas, i))
                vocab[token] = index
    return vocab


class Tokenizer:
    def __init__(self, vocab_file: str, has_index: bool, unk_token='[UNK]'):
        if not os.path.exists(vocab_file):
            raise ValueError("Can't Find Vocabulary File %s" % vocab_file)
        self.unk_token = unk_token
        self.vocab = load_vocab(vocab_file, has_index=has_index)
        self.reverse_vocab = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])

    def convert_tokens_to_ids(self, tokens: Union[List[str], str]) -> Union[int, List[int]]:
        if tokens is None:
            return None
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        ids = list()
        for token in tokens:
            ids.append(self._convert_token_to_id(token))
        return ids

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token)

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[int, List[int]]:
        if ids is None:
            return None
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        tokens = list()
        for id in ids:
            tokens.append(self._convert_id_to_token(id))
        return tokens

    def _convert_id_to_token(self, id: int) -> str:
        return self.reverse_vocab.get(id, -1)
