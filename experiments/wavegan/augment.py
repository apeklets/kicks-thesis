import logging
import librosa
import numpy as np
import pescador
from librosa.effects import trim
from librosa.util import normalize
import sys
sys.path.append("../../notebooks")
from quality.decay import env_qual, rms_qual
from quality.response import res_qual
from quality.tonality import frq_qual, ihd_qual
from quality.filters import butter_hp, butter_lp

from sample import create_batch_generator

LOGGER = logging.getLogger("wavegan")
LOGGER.setLevel(logging.DEBUG)


def load_sample(path, fs=44100):
    sample, _ = librosa.load(path, sr=fs)
    return normalize(butter_hp(trim(sample, top_db=50)[0], 10))


def sort_low(path):
    sample = load_sample(path)
    return 5 * env_qual(sample) + rms_qual(sample) + frq_qual(sample)


def sort_high(path):
    sample = load_sample(path)
    return res_qual(sample) + ihd_qual(sample)


def layer_lowhigh(sample1, sample2, freq=400):
    lp = butter_lp(sample1, freq, order=3)
    hp = butter_hp(sample2, freq, order=4)
    n = max(len(lp), len(hp))
    out = np.empty(n)
    for i in range(0, min(len(lp), len(hp))):
        out[i] = lp[i] + hp[i]
    for i in range(min(len(lp), len(hp)), n):
        if i >= len(lp):
            out[i] = hp[i]
        else:
            out[i] = lp[i]
    return normalize(out)


def layer_samples(path1, path2, window_length=32768, fs=44100):
    """
    Audio sample generator
    """
    try:
        if path2 is not None:
            sample1 = load_sample(path1, fs=fs)
            sample2 = load_sample(path2, fs=fs)
            audio_data = layer_lowhigh(sample1, sample2, 400)
        else:
            audio_data = load_sample(path1, fs=fs)
    except Exception as e:
        LOGGER.error("Could not load {} or {}: {}".format(path1, path2, str(e)))
        raise StopIteration()

    audio_len = len(audio_data)

    # Pad audio to at least a single frame
    if audio_len < window_length:
        pad_length = window_length - audio_len
        left_pad = 0  # we want all kicks to start at 0
        right_pad = pad_length - left_pad

        audio_data = np.pad(audio_data, (left_pad, right_pad), mode="constant")
        audio_len = len(audio_data)

    if audio_len == window_length:
        # If we only have a single frame's worth of audio, just yield the whole audio
        sample = audio_data
    else:
        # Sample a random window from the audio file
        # start_idx = np.random.randint(0, audio_len - window_length)
        start_idx = 0  # start at the start
        end_idx = start_idx + window_length
        sample = audio_data[start_idx:end_idx]

        if sample[end_idx - 1] != 0.0:  # apply some declicking
            h = np.hanning(513)
            for i in range(1, 256):
                sample[end_idx - i] *= h[513 - i]

    sample = sample.astype("float32")
    assert not np.any(np.isnan(sample))

    return sample


def augmented_batch_gen(dataset, batch_size):
    N = len(dataset)
    LOGGER.info(f"Found {N} input samples. Sorting...")
    dataset.sort(key=sort_low)
    top_low = dataset[0 : N // 2]
    dataset.sort(key=sort_high)
    top_high = dataset[0 : N // 2]

    def gen_layers(top_high, path1):
        for path2 in top_high:
            yield {'X': layer_samples(path1, path2)}

    streamers = []
    for path1 in top_low:
        s = pescador.Streamer(gen_layers, top_high, path1)
        streamers.append(s)
    mux = pescador.ShuffledMux(streamers)
    batch_gen = pescador.buffer_stream(mux, batch_size)
    return batch_gen


def create_augm_data_split(
    audio_filepath_list,
    valid_ratio,
    test_ratio,
    train_batch_size,
    valid_size,
    test_size,
):
    num_files = len(audio_filepath_list)
    num_valid = int(np.ceil(num_files * valid_ratio))
    num_test = int(np.ceil(num_files * test_ratio))
    num_train = num_files - num_valid - num_test

    assert num_valid > 0
    assert num_test > 0
    assert num_train > 0

    valid_files = audio_filepath_list[:num_valid]
    test_files = audio_filepath_list[num_valid : num_valid + num_test]
    train_files = audio_filepath_list[num_valid + num_test :]

    train_gen = augmented_batch_gen(train_files, train_batch_size)
    valid_data = next(iter(create_batch_generator(valid_files, valid_size)))
    test_data = next(iter(create_batch_generator(test_files, test_size)))

    return train_gen, valid_data, test_data
