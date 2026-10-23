"""Energy regions and representative waveform slices, adapted from ../metta.

Boundaries are seconds, with no BPM, beat, bar, or section-label assumptions.
"""

import math

import numpy as np
from numpy.typing import NDArray


def energy_regions(audio: NDArray[np.float32], sample_rate: int) -> list[tuple[float, float]]:
    """Find sustained energy changes using one-second RMS bins.

    Carry over Metta's normalized energy contrast and minimum-region approach.
    Scan the whole waveform on CPU; MuQ only sees the selected short slices.
    """
    if sample_rate <= 0 or audio.ndim != 1 or not audio.size:
        raise ValueError("Expected nonempty mono audio and a positive sample rate.")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or infinity.")
    duration = len(audio) / sample_rate
    energy = np.array(
        [
            np.sqrt(np.mean(np.square(audio[start : start + sample_rate], dtype=np.float64)))
            for start in range(0, len(audio), sample_rate)
        ]
    )
    spread = float(np.ptp(energy))
    if spread <= 1e-10:
        return [(0.0, duration)]
    energy = ((energy - energy.min()) / spread) ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(energy)))
    # ponytail: seconds-based heuristic; replace boundary detection with a verified
    # beat/downbeat grid later, keeping the same (start_seconds, end_seconds) output.
    minimum = 16
    lookahead = 4
    boundaries = [0]
    for index in range(minimum, len(energy) - lookahead + 1):
        start = boundaries[-1]
        if index - start < minimum:
            continue
        previous = (cumulative[index] - cumulative[start]) / (index - start)
        following = (cumulative[index + lookahead] - cumulative[index]) / lookahead
        if abs(following - previous) > 0.15:
            # Place the boundary at the strongest change in the lookahead, so
            # an early trigger does not split a single sustained step twice.
            changes = np.abs(np.diff(energy[index - 1 : index + lookahead]))
            boundaries.append(index + int(np.argmax(changes)))
    # Merge a short final region into its predecessor.
    if len(boundaries) > 1 and duration - boundaries[-1] < minimum:
        boundaries.pop()
    return list(zip([float(value) for value in boundaries], [*boundaries[1:], duration]))


def select_slices(
    audio: NDArray[np.float32],
    sample_rate: int,
    regions: list[tuple[float, float]],
    slice_duration: float = 30,
) -> list[tuple[float, float, str]]:
    """Select a start window and, when distinct, a window around peak RMS.

    Windows stay within their region and never exceed slice_duration. Regions
    shorter than one second are skipped. No audio buffers are copied here.
    """
    if not math.isfinite(slice_duration) or slice_duration < 1:
        raise ValueError("Slice duration must be finite and at least 1 second.")
    if sample_rate <= 0 or audio.ndim != 1 or not audio.size:
        raise ValueError("Expected nonempty mono audio and a positive sample rate.")
    slices: list[tuple[float, float, str]] = []
    for start, end in regions:
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError("Region boundaries must be finite, ordered, and nonnegative.")
        if end > len(audio) / sample_rate:
            raise ValueError("Region extends beyond the audio.")
        if end - start < 1:
            continue
        slices.append((start, min(end, start + slice_duration), "start"))
        if end - start < slice_duration * 1.5:
            continue
        first, last = round(start * sample_rate), round(end * sample_rate)
        starts = range(first, last, sample_rate)
        energies = [
            float(np.mean(np.square(audio[index : min(index + sample_rate, last)])))
            for index in starts
        ]
        peak = (starts[int(np.argmax(energies))] + sample_rate / 2) / sample_rate
        peak_start = max(start, min(end - slice_duration, peak - slice_duration / 2))
        if peak_start >= start + slice_duration * 0.8:
            slices.append((peak_start, peak_start + slice_duration, "peak"))
    return slices
