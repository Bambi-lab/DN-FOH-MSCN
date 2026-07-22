# Data

ShipsEar audio is not included. Obtain it from its official distributor and
comply with its license. The executable simulation stage expects already
prepared five-class, mono WAV segments under `A/`--`E/`, sampled at 12 kHz
and 3 s long. The local project contains 16,898 such segments, but the exact
original-nine-class to five-class mapping and the augmentation RNG seed are
not fully recorded; see `KNOWN_LIMITATIONS.md`.

The released split manifests contain only relative sample identifiers,
labels, recording identifiers, and split assignments. They contain no audio.
