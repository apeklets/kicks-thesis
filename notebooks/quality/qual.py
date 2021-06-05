from os import listdir

from librosa import load
from librosa.effects import trim
from librosa.util import normalize

from decay import env_qual, rms_qual
from response import res_qual
from tonality import frq_qual, ihd_qual
from filters import butter_hp


def kick_qual(sample, weights: list = [10, 0.5, 1, 2, 0.5]):
    """
    Quality = weights[0] * env_qual + weights[1] * res_qual + weights[2] * frq_qual + weights[3] * ihd_qual + weights[4] * rms_qual
    """
    # preprocessing (trim -> normalize -> highpass)
    sample = normalize(butter_hp(trim(sample, top_db=50)[0], 10))
    return (
        weights[0] * env_qual(sample)
        + weights[1] * res_qual(sample)
        + weights[2] * frq_qual(sample)
        + weights[3] * ihd_qual(sample)
        + weights[4] * rms_qual(sample)
    )


def eval_dir(path, weights: list = [10, 0.5, 1, 2, 0.5]):
    quals = []
    mean = 0
    files = listdir(path)
    for f in files:
        sample, _ = load(path + "/" + f)
        q = kick_qual(sample)
        quals.append((f, q))
        mean += q
    mean = mean / len(files)
    return mean, quals
