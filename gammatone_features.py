"""
Gammatone frequency spectrum feature extraction
Paper Sec. 3.4 Gammatone filterbank + Sec. 3.5.1 Gammatone frequency spectrum features
"""
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Set Chinese font (for backward compatibility with labels)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== Paper parameters ==========
SR = 12000              # Sample rate
PRE_EMPHASIS = 0.97     # Pre-emphasis coefficient (Eq. 3.16)
FRAME_LENGTH_MS = 85    # Frame length 85 ms
FRAME_SHIFT_MS = 21     # Frame shift 21 ms
N_FILTERS = 128         # Number of Gammatone filter channels
F_LOW = 50              # Lowest frequency (Hz)
F_HIGH = 6000           # Highest frequency (Hz)
FILTER_ORDER = 4        # Filter order n = 4
B_COEFF = 1.019         # Bandwidth coefficient B

# ========== ERB-related functions (Eq. 3.13, 3.14) ==========
def hz_to_erb(f):
    """Convert actual frequency to ERB frequency (Eq. 3.13)"""
    return 21.4 * np.log10(4.37 * f / 1000 + 1)

def erb_to_hz(e):
    """Convert ERB frequency to actual frequency"""
    return (10 ** (e / 21.4) - 1) * 1000 / 4.37

def erb_bandwidth(fc):
    """ERB critical bandwidth (Eq. 3.14)"""
    return 24.7 + 0.108 * fc

# ========== Gammatone filterbank ==========
def gammatone_filterbank(n_filters, nfft, sr, f_low, f_high):
    """Build Gammatone filterbank frequency response matrix

    Frequency-domain representation based on Eq. 3.15:
    |H(f)|^2 = 1 / (1 + ((f - fc) / b)^2)^n
    where b = B * ERB(fc), n = 4
    """
    # Center frequencies uniformly spaced on ERB scale
    erb_low = hz_to_erb(f_low)
    erb_high = hz_to_erb(f_high)
    erb_points = np.linspace(erb_low, erb_high, n_filters)
    center_freqs = erb_to_hz(erb_points)

    # Frequency axis
    freqs = np.linspace(0, sr / 2, nfft // 2 + 1)

    # Build filterbank
    filterbank = np.zeros((n_filters, len(freqs)))

    for i in range(n_filters):
        fc = center_freqs[i]
        b = B_COEFF * erb_bandwidth(fc)
        # Gammatone magnitude response (n = 4th order)
        filterbank[i, :] = 1 / (1 + ((freqs - fc) / b) ** 2) ** FILTER_ORDER

    return filterbank, center_freqs

# ========== Gammatone frequency spectrum feature extraction ==========
def extract_gammatone_spectrum(signal, sr=SR):
    """Extract Gammatone frequency spectrum features (pipeline from Sec. 3.5.1 of paper)"""
    # (1) Pre-emphasis (Eq. 3.16)
    emphasized = np.append(signal[0], signal[1:] - PRE_EMPHASIS * signal[:-1])

    # (2) Frame blocking
    frame_length = int(FRAME_LENGTH_MS * sr / 1000)
    frame_shift = int(FRAME_SHIFT_MS * sr / 1000)

    n_frames = 1 + (len(emphasized) - frame_length) // frame_shift
    frames = np.zeros((n_frames, frame_length))
    for i in range(n_frames):
        start = i * frame_shift
        frames[i] = emphasized[start:start + frame_length]

    # (3) Hamming windowing (Eq. 3.17)
    hamming = np.hamming(frame_length)
    frames *= hamming

    # (4) FFT (Eq. 3.18)
    nfft = 2 ** int(np.ceil(np.log2(frame_length)))
    fft_result = np.fft.rfft(frames, n=nfft)

    # (5) Power spectrum (squared magnitude)
    power_spectrum = np.abs(fft_result) ** 2

    # (6) Gammatone frequency filtering (Eq. 3.19)
    filterbank, center_freqs = gammatone_filterbank(N_FILTERS, nfft, sr, F_LOW, F_HIGH)
    gammatone_spec = np.dot(power_spectrum, filterbank.T)

    # (7) Log compression
    gammatone_spec = np.log(gammatone_spec + 1e-10)

    return gammatone_spec.T, center_freqs  # Transpose: (freq, time)

# ========== Visualization ==========
def visualize_gammatone_spectrum_all(dataset_dir, output_path):
    """Visualize Gammatone frequency spectra for five classes in style of Fig. 3.9"""
    dataset_dir = Path(dataset_dir)
    categories = ['A', 'B', 'C', 'D', 'E']
    labels = ['(a) Class A', '(b) Class B', '(c) Class C', '(d) Class D', '(e) Class E']

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0]]
    axes[2, 1].axis('off')

    for idx, (cat, label) in enumerate(zip(categories, labels)):
        wav_files = sorted((dataset_dir / cat).glob('*.wav'))
        if len(wav_files) == 0:
            continue
        signal, sr = librosa.load(wav_files[0], sr=SR)
        spec, center_freqs = extract_gammatone_spectrum(signal, sr)

        ax = axes_flat[idx]
        n_frames = spec.shape[1]
        im = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis',
                       extent=[0, n_frames, 0, N_FILTERS])
        ax.set_xlabel('Frames')
        ax.set_ylabel('Channels')
        ax.set_title(label)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == '__main__':
    dataset_dir = './data/shipsear_segments'
    output_dir = Path('./figures/gammatone_spectrum')
    output_dir.mkdir(parents=True, exist_ok=True)

    visualize_gammatone_spectrum_all(dataset_dir, output_dir / 'gammatone_spectrum_all.png')
