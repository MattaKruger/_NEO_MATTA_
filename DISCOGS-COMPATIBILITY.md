# Discogs classifier platform feasibility

Checked 2026-09-08. Research only; no classifier installation or inference was performed.

## Direct transfer is the first option

The old project has two distinct classifier paths: `discogs_genre_extractor.py`
uses Discogs Effnet for track labels, while `essentia_semantic_extractor.py` uses
MAEST for slice labels. Both run through Essentia's TensorFlow algorithms.

**Apple Silicon does have a published Python 3.11 wheel.**
`essentia-tensorflow==2.1b6.dev1389` publishes
`cp311-cp311-macosx_15_0_arm64` (115.3 MB). The newer `dev1438` release currently
publishes Python 3.14 wheels only. Pin the compatible version for an initial
experiment. The inspected machine reports macOS 26.5.2, above the wheel's minimum.
[PyPI release files](https://pypi.org/project/essentia-tensorflow/2.1b6.dev1389/#files),
[latest release](https://pypi.org/project/essentia-tensorflow/).

This establishes packaging availability, **not successful runtime compatibility**.
Before changing project dependencies, test the wheel with the POC's NumPy 2.x,
check both algorithm imports, and run a real audio window with the cached graphs.
Assume CPU for this additional inference; availability of a native wheel does not
establish MPS acceleration. Essentia's documented TensorFlow wrapper is separate
from the POC's PyTorch/MPS execution.
[Official installation guidance](https://essentia.upf.edu/machine_learning.html).

## Preserve model contracts

Effnet's official metadata specifies 16 kHz audio, 400 prediction values from
`PartitionedCall:0`, and a separate 1280-dimensional embedding output from
`PartitionedCall:1`. Predictions already have sigmoid activation. Resample a copy
of the selected window to 16 kHz; never feed the MuQ 24 kHz waveform unchanged.
Keep predicted labels/scores separate from the MuQ vector column.
[Model metadata](https://essentia.upf.edu/models/music-style-classification/discogs-effnet/discogs-effnet-bs64-1.json).

The Effnet graph declares a fixed batch of 64, so do not blindly reduce its batch
size to the MuQ batch size of one. Essentia supports padding a final batch and
returning predictions corresponding to real patches. Its default final incomplete
patch is discarded; very short slices need an explicit tested policy.
[Algorithm reference](https://essentia.upf.edu/reference/std_TensorflowPredictEffnetDiscogs.html).

## Alternative if the direct runtime fails

The original MAEST author supplies a native PyTorch implementation with 16 kHz
waveform input, checkpoint loading, and label prediction. This is a MAEST
alternative, **not a drop-in replacement for Effnet**. Upstream documents macOS
testing, but does not establish this POC's MPS compatibility. Its convenience
label method averages sigmoid predictions; compare that carefully with the old
project's sigmoid of averaged chunk logits if preserving scores matters.
[Author's implementation](https://github.com/palonso/MAEST).

Its package currently pins NumPy `~=1.26` and includes training dependencies such
as Lightning and Sacred, conflicting with the POC's NumPy 2.x requirement. Using
it unchanged would therefore require dependency changes or an isolated environment.
Do not add this complexity before trying the available Essentia wheel.
[Upstream dependency declarations](https://raw.githubusercontent.com/palonso/MAEST/main/pyproject.toml).

Recommended next check: isolated uv environment, exact Essentia wheel, existing
model files, one real 16 kHz window. Confirm finite output shape and class ordering,
then integrate one classifier; leave alternative backends until a measured blocker
requires one. No Docker inference worker or custom TensorFlow build is currently
justified.
