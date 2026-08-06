# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Bootstrap pinned LM Evaluation Harness with narrow IrokoBench compatibility fixes."""

from __future__ import annotations


def install_iroko_compatibility() -> None:
    """Expose the harness's intended weighted-F1 aggregator to Iroko task YAMLs."""
    from lm_eval.tasks import _yaml_loader
    from lm_eval.utils import weighted_f1_score

    original_import = _yaml_loader._import_func_in_yml

    def import_with_iroko_fallback(spec: str, base_dir: str):
        try:
            return original_import(spec, base_dir)
        except AttributeError:
            if spec.rsplit(".", maxsplit=1)[-1] == "weighted_f1_score":
                return weighted_f1_score
            raise

    _yaml_loader._import_func_in_yml = import_with_iroko_fallback


def main() -> None:
    """Install compatibility symbols and delegate to the upstream CLI."""
    install_iroko_compatibility()
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":
    main()
