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

"""Data loading helpers."""

import tensorflow as tf
import tensorflow_datasets as tfds


@tfds.decode.make_decoder()
def decode_and_central_square_crop(img: tf.Tensor, feature) -> tf.Tensor:
  """Crop to square along the long edge."""
  del feature  # Unused.
  h, w, _ = tf.unstack(tf.io.extract_jpeg_shape(img))
  box = tf.where(h > w, [h // 2 - w // 2, 0, w, w], [0, w // 2 - h // 2, h, h])
  return tf.image.decode_and_crop_jpeg(img, box, channels=3)
