# Copyright © 2023 Apple Inc.

"""
Code to preprocess the WikiSQL dataset adapted from
https://github.com/salesforce/WikiSQL and
https://huggingface.co/sqllama/sqllama-V0/blob/main/wikisql.ipynb .
"""


import json
import os


def load():
    """
    Load all three splits of the WikiSQL dataset.
    """
    return (WikiSQL(dn) for dn in ["train", "dev", "test"])


class WikiSQL:
    def __init__(self, dataset, save_dir="/tmp"):
        valid_sets = ("train", "dev", "test")
        if dataset not in valid_sets:
            raise ValueError(f"Dataset must be in {valid_sets}, got {dataset}")
        data_dir = os.path.join(save_dir, "wikisql")
        self._maybe_download(data_dir)

        self._parse_tables(os.path.join(data_dir, f"data/{dataset}.tables.jsonl"))
        self._parse_queries(os.path.join(data_dir, f"data/{dataset}.jsonl"))

    def _maybe_download(self, data_dir):
        if not os.path.exists(data_dir):
            import io
            import tarfile
            from urllib import request

            url = "https://raw.githubusercontent.com/salesforce/WikiSQL/master/data.tar.bz2"
            r = request.urlopen(url)
            with tarfile.open(fileobj=io.BytesIO(r.read())) as tf:
                tf.extractall(data_dir)

    def _parse_tables(self, tables):
        self._tables = {}
        with open(tables) as f:
            for line in f:
                table = json.loads(line)
                self._tables[table["id"]] = {
                    "columns": table["header"],
                    "types": table["types"],
                    "desc": f"table: {table['id']}\ncolumns: {', '.join(table['header'])}",
                }

    def _parse_queries(self, queries):
        self._queries = []
        with open(queries) as f:
            for line in f:
                query = json.loads(line)
                table = self._tables[query["table_id"]]
                question = query["question"]
                answer = self.query_to_text(
                    query["sql"], query["table_id"], table["columns"], table["types"]
                )
                self._queries.append(
                    f"<s>{table['desc']}\nQ: {question}\nA: {answer}</s>"
                )

    def query_to_text(self, query, table, columns, types):
        aggregation_ops = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
        condition_ops = ["=", ">", "<", "OP"]
        column = columns[query["sel"]]
        aggregation = (aggregation_ops[query["agg"]] + " ") if query["agg"] > 0 else ""
        sql = f"SELECT {aggregation}{column} FROM {table}"

        conditions = query["conds"]
        if conditions:
            cs = []
            for i, o, v in conditions:
                column = columns[i]
                op = condition_ops[o]

                if types[i] == "text":
                    value = f"'{v}'"
                else:
                    value = v
                cs.append(f"{column} {op} {value}")

            sql += " WHERE " + " AND ".join(cs)

        return sql

    def __getitem__(self, idx):
        return self._queries[idx]

    def __len__(self):
        return len(self._queries)


if __name__ == "__main__":
    datanames = ["train", "dev", "test"]
    sizes = [56355, 8421, 15878]
    for dataname, size in zip(datanames, sizes):
        len(WikiSQL(dataname)) == size, f"Wrong {dataname} set size."

    # Write the sets to jsonl
    import json

    train, dev, test = load()
    datasets = [
        (train, "train", 1000),
        (dev, "valid", 100),
        (test, "test", 100),
    ]
    for dataset, name, size in datasets:
        with open(f"data/{name}.jsonl", "w") as fid:
            for e, t in zip(range(size), dataset):
                # Strip the <s>, </s> since the tokenizer adds them
                json.dump({"text": t[3:-4]}, fid)
                fid.write("\n")
