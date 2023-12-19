# Copyright © 2023 Apple Inc.

import io
import itertools
import os
import zipfile
from urllib import request

import numpy as np


def load_dataset(dataname):
    if dataname == "ptb":
        return ptb()
    elif dataname == "wikitext2":
        return wikitext(dataset="2")
    else:
        return wikitext(dataset="103")


def _load(save_dir, filenames):
    # *NB* First file is expected to be the training set
    with open(os.path.join(save_dir, filenames[0]), "r") as fid:
        vocab = set(t for l in fid.readlines() for t in l.strip().split(" "))
    eos = "<eos>"
    vocab.add(eos)
    vocab = {v: i for i, v in enumerate(vocab)}

    def to_array(dataset):
        with open(os.path.join(save_dir, dataset), "r") as fid:
            lines = (l.strip().split(" ") for l in fid.readlines())
        return np.array(
            [vocab[w] for line in lines for w in itertools.chain(line, [eos])],
            dtype=np.uint32,
        )

    datasets = [to_array(fn) for fn in filenames]
    return vocab, *datasets


def wikitext(dataset="2", save_dir="/tmp"):
    """
    Load the WikiText-* language modeling dataset:
        https://paperswithcode.com/dataset/wikitext-2
        https://paperswithcode.com/dataset/wikitext-103

    """
    if dataset not in ("2", "103"):
        raise ValueError(f'Dataset must be either "2" or "103", got {dataset}')

    filenames = ["wiki.train.tokens", "wiki.valid.tokens", "wiki.test.tokens"]
    dataname = f"wikitext-{dataset}"
    data_dir = os.path.join(save_dir, dataname)
    if not os.path.exists(data_dir):
        base_url = "https://s3.amazonaws.com/research.metamind.io/wikitext/"
        zip_file_url = base_url + dataname + "-v1.zip"
        r = request.urlopen(zip_file_url)
        with zipfile.ZipFile(io.BytesIO(r.read())) as zf:
            zf.extractall(save_dir)

    return _load(data_dir, filenames)


def ptb(save_dir="/tmp"):
    """
    Load the PTB language modeling dataset:
        https://paperswithcode.com/dataset/penn-treebank
    """
    filenames = [
        "ptb.train.txt",
        "ptb.valid.txt",
        "ptb.test.txt",
    ]

    def download_and_save(save_dir):
        base_url = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/"
        for name in filenames:
            out_file = os.path.join(save_dir, name)
            if not os.path.exists(out_file):
                request.urlretrieve(base_url + name, out_file)

    save_dir = os.path.join(save_dir, "ptb")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    download_and_save(save_dir)

    return _load(save_dir, filenames)


if __name__ == "__main__":
    vocab, train, val, test = ptb()
    assert len(vocab) == 10000, "PTB: Wrong vocab size"

    vocab, train, val, test = wikitext()
    assert len(vocab) == 33279, "WikiText: Wrong vocab size"
