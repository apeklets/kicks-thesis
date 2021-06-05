import os
import librosa
import soundfile

def save_samples(epoch_samples, epoch, output_dir, fs=44100):
    """
    Save output samples to disk
    """
    sample_dir = os.path.join(output_dir, str(epoch))
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    for idx, samp in enumerate(epoch_samples):
        output_path = os.path.join(sample_dir, "{}.wav".format(idx+1))
        samp = samp[0]
        # librosa.output.write_wav(output_path, samp, fs)
        soundfile.write(output_path, samp, fs)
