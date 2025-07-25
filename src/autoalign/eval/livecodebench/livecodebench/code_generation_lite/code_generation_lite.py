"""
Modified from the original code at https://huggingface.co/datasets/codeparrot/apps/blob/main/apps.py
"""

import json
import datasets


_REPO_NAME = "loubnabnl/apps"

_CITATION = """\
@article{jain2024livecodebench,
    title={LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code},
    author={Jain, Naman and Han, King and Gu, Alex and Li, Wen-Ding and Yan, Fanjia and Zhang, Tianjun and Wang, Sida and Solar-Lezama, Armando and Sen, Koushik and Stoica, Ion},
    journal={arXiv preprint arXiv:2403.07974},
    year={2024}
}
"""

_DESCRIPTION = """\
LiveCodeBench is a temporaly updating benchmark for code generation. Please check the homepage: https://livecodebench.github.io/.
"""

_HOMEPAGE = "https://livecodebench.github.io/"
_URLS = {
    "train": [],
    "test": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ],
}
ALLOWED_FILES = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
    ],
    "release_v6": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ],
    "release_latest": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ],
}
v_list = ["v1", "v2", "v3", "v4", "v5", "v6"]
for v in v_list:
    ALLOWED_FILES[v] = [f"test{v[1:]}.jsonl" if v != "v1" else "test.jsonl"]

n_vs = len(v_list)
for idx1 in range(1, n_vs + 1):
    for idx2 in range(idx1 + 1, n_vs + 1):
        ALLOWED_FILES[v_list[idx1 - 1] + "_" + v_list[idx2 - 1]] = [
            f"test{idx}.jsonl" if idx != 1 else "test.jsonl"
            for idx in range(idx1, idx2 + 1)
        ]


_VERSIONS = list(ALLOWED_FILES.keys())
_VERSIONS_CONFIGS = _VERSIONS


class LCBCodeGenConfig(datasets.BuilderConfig):
    """BuilderConfig for the LCBCodeGenConfig dataset."""

    def __init__(self, *args, version_tag="release_latest", **kwargs):
        """BuilderConfig for the LCBCodeGenConfig dataset.
        Args:
            version (:obj:`List[str]`): The version of the dataset to use (only single length lists are supports).
            **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(
            *args,
            name=version_tag,
            **kwargs,
        )

        assert (
            version_tag in _VERSIONS_CONFIGS
        ), f"{version_tag} not in {_VERSIONS_CONFIGS}."

        self.version_tag = version_tag


class LCBCodeGen(datasets.GeneratorBasedBuilder):
    """LCBCodeGen dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = LCBCodeGenConfig
    BUILDER_CONFIGS = [
        LCBCodeGenConfig(version_tag=version) for version in _VERSIONS_CONFIGS
    ]
    DEFAULT_CONFIG_NAME = "release_latest"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "question_title": datasets.Value("string"),
                    "question_content": datasets.Value("string"),
                    "platform": datasets.Value("string"),
                    "question_id": datasets.Value("string"),
                    "contest_id": datasets.Value("string"),
                    "contest_date": datasets.Value("string"),
                    "starter_code": datasets.Value("string"),
                    "difficulty": datasets.Value("string"),
                    "public_test_cases": datasets.Value("string"),
                    "private_test_cases": datasets.Value("string"),
                    "metadata": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
            homepage=_HOMEPAGE,
            license="MIT License",
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "file_paths": downloaded_files["test"],
                    "file_names": _URLS["test"],
                },
            ),
        ]

    def _generate_examples(self, file_paths, file_names):
        key = 0
        for file_path, file_name in zip(file_paths, file_names):
            if file_name not in ALLOWED_FILES[self.config.version_tag]:
                continue
            for idx, line in enumerate(open(file_path, "r")):
                line_data = json.loads(line)
                yield key, line_data
                key += 1
