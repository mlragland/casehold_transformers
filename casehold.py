# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CaseHOLD dataset loading script"""


import csv
import datasets


_CITATION = """\
@inproceedings{zhengguha2021,
    title={When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset},
    author={Lucia Zheng and Neel Guha and Brandon R. Anderson and Peter Henderson and Daniel E. Ho},
    year={2021},
    eprint={2104.08671},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    booktitle={Proceedings of the Eighteenth International Conference on Artificial Intelligence and Law},
    publisher={Association for Computing Machinery}
}
"""

_DESCRIPTION = """\
CaseHOLD (Case Holdings On Legal Decisions) is a law dataset comprised of over 53,000+ multiple choice questions to identify the relevant holding of a cited case.
"""

_HOMEPAGE = "https://reglab.stanford.edu/data/casehold-benchmark/"

_LICENSE = ""

_PATHS = {
    "all": {
        "train": "data/all/train.csv",
        "val": "data/all/val.csv",
        "test": "data/all/test.csv"
    },
    "fold_1": {
        "train": "data/fold_1/train.csv",
        "val": "data/fold_1/val.csv",
        "test": "data/fold_1/test.csv"
    },
    "fold_2": {
        "train": "data/fold_2/train.csv",
        "val": "data/fold_2/val.csv",
        "test": "data/fold_2/test.csv"
    },
    "fold_3": {
        "train": "data/fold_3/train.csv",
        "val": "data/fold_3/val.csv",
        "test": "data/fold_3/test.csv"
    },
    "fold_4": {
        "train": "data/fold_4/train.csv",
        "val": "data/fold_4/val.csv",
        "test": "data/fold_4/test.csv"
    },
    "fold_5": {
        "train": "data/fold_5/train.csv",
        "val": "data/fold_5/val.csv",
        "test": "data/fold_5/test.csv"
    },
    "fold_6": {
        "train": "data/fold_6/train.csv",
        "val": "data/fold_6/val.csv",
        "test": "data/fold_6/test.csv"
    },
    "fold_7": {
        "train": "data/fold_7/train.csv",
        "val": "data/fold_7/val.csv",
        "test": "data/fold_7/test.csv"
    },
    "fold_8": {
        "train": "data/fold_8/train.csv",
        "val": "data/fold_8/val.csv",
        "test": "data/fold_8/test.csv"
    },
    "fold_9": {
        "train": "data/fold_9/train.csv",
        "val": "data/fold_9/val.csv",
        "test": "data/fold_9/test.csv"
    },
    "fold_10": {
        "train": "data/fold_10/train.csv",
        "val": "data/fold_10/val.csv",
        "test": "data/fold_10/test.csv"
    },
}

class CaseHOLD(datasets.GeneratorBasedBuilder):
    """Multiple choice law dataset to identify the relevant holding of a cited case."""

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="all", version=VERSION, description="This is the train/val/test split on all of the data."),
        datasets.BuilderConfig(name="fold_1", version=VERSION, description="This is the fold 1 train/val/test split for 10-fold CV on the data."),
        datasets.BuilderConfig(name="fold_2", version=VERSION, description="This is the fold 2 train/val/test split for 10-fold CV on the data."),
        datasets.BuilderConfig(name="fold_3", version=VERSION, description="This is the fold 3 train/val/test split for 10-fold CV on the data."),
        datasets.BuilderConfig(name="fold_4", version=VERSION, description="This is the fold 4 train/val/test split for 10-fold CV on the data."),
        datasets.BuilderConfig(name="fold_5", version=VERSION, description="This is the fold 5 train/val/test split for 10-fold CV on the data."),
        datasets.BuilderConfig(name="fold_6", version=VERSION, description="This is the fold 6 train/val/test split for 10-fold CV on the data."),
        datasets.BuilderConfig(name="fold_7", version=VERSION, description="This is the fold 7 train/val/test split for 10-fold CV on the data."),
        datasets.BuilderConfig(name="fold_8", version=VERSION, description="This is the fold 8 train/val/test split for 10-fold CV on the data."),
        datasets.BuilderConfig(name="fold_9", version=VERSION, description="This is the fold 9 train/val/test split for 10-fold CV on the data."),
        datasets.BuilderConfig(name="fold_10", version=VERSION, description="This is the fold 10 train/val/test split for 10-fold CV on the data."),
    ]

    DEFAULT_CONFIG_NAME = "all"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        features = datasets.Features(
            {
                "example_id": datasets.Value("int32"),
                "citing_prompt": datasets.Value("string"),
                "holding_0": datasets.Value("string"),
                "holding_1": datasets.Value("string"),
                "holding_2": datasets.Value("string"),
                "holding_3": datasets.Value("string"),
                "holding_4": datasets.Value("string"),
                "label": datasets.Value("string")
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        
        urls = _PATHS[self.config.name]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir["val"],
                    "split": "val",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir["test"],
                    "split": "test"
                },
            ),
        ]

    def _generate_examples(
        self, filepath, split  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """ Yields examples as (key, example) tuples. """
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.

        with open(filepath, encoding="utf-8") as f:
            # Add some logic for the csv reader here
            csv_reader = csv.reader(f)
            # Skip column names row
            next(csv_reader)

            for id_, row in enumerate(csv_reader):
                yield id_, {
                    "example_id": int(row[0]),
                    "citing_prompt": str(row[1]),
                    "holding_0": str(row[2]),
                    "holding_1": str(row[3]),
                    "holding_2": str(row[4]),
                    "holding_3": str(row[5]),
                    "holding_4": str(row[6]),
                    "label": str(row[12]),
                }

