# Copyright 2025 Hackable Diffusion Authors.
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

"""Init file for hackable_diffusion.lib."""

# Do NOT add anything here !!
# Indeed, top-level `__init__.py` makes it hard to import a specific sub-module
# without triggering a full import of the codebase.
# Instead, the public API is exposed in `hd.py`


def __getattr__(name: str):  # pylint: disable=invalid-name
  """Catches `import hackable_diffusion.lib as hd` errors."""
  del name
  raise AttributeError(
      'Please always use "from hackable_diffusion import hd", '
      'never "import hackable_diffusion.lib as hd".'
  )
