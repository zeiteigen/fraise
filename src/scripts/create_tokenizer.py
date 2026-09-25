from tokenizer import load_tokenizer
from config import get_dataset_loader


def main():
    VOCAB_SIZE = input("Enter the desired vocabulary size (default: 8192): ")
    if not VOCAB_SIZE:
        VOCAB_SIZE = 8192
    else:
        VOCAB_SIZE = int(VOCAB_SIZE)
    load_tokenizer(vocab_size=VOCAB_SIZE, dataset=get_dataset_loader())
