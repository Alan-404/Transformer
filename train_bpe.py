import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from typing import List

import fire

def train(text_paths: List[str],
          saved_path: str):
    if os.path.exists(saved_path) == False:
        os.makedirs(saved_path)
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(files=text_paths, trainer=trainer)

    tokenizer.save(saved_path)

if __name__ == '__main__':
    fire.Fire(train)