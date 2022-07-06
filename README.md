# PointCloudTransformers

This repository is forked from [qq456cvb/Point-Transformers](https://github.com/qq456cvb/Point-Transformers), and I made the following changes to the code related to the classification task.

## 1. To adapt `Hydar 1.1 & 1.2`

- Delete the first line of `config/model/*.yaml` file: `# @package _group_` (this syntax is no longer supported in newer versions of Hydra)

- `cls.yaml` Modify:
  - `defaults` added `_self_` (required for higher versions)
  - Add `hydra.job.chdir = Ture` (higher versions already have this property set to `False` by default)
  - Add `custom_output_dir` to allow users to customize the output path, so that the same task can be run multiple times.

## 2. Added `test_cls.py`

- The purpose of adding this script is to perform deep learning testing and validation experiments separately.
- We conducted test experiments on the ModelNet40 dataset and the program works fine.
- Run with `python test_cls.py`.

## Notes

For other relevant reference details, please check the original repository [qq456cvb/Point-Transformers](https://github.com/qq456cvb/Point-Transformers).
