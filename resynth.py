import os
import numpy as np
import matplotlib.pyplot as plt
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import fastgen
from IPython.display import Audio


def load_encoding(fname, sample_length=None, sr=16000,
                  ckpt='model/model.ckpt-200000'):
    """Load audio encoding. Returns tuple."""
    audio = utils.load_audio(fname, sample_length=sample_length, sr=sr)
    encoding = fastgen.encode(audio, ckpt, sample_length)
    return audio, encoding

# set sample rate and sample length, then load audio file.
# TODO: try with higher sample rate.
# Used mp3 file and low rate due to not enough compute power
sr = 16000
sample_length = 3200000
aud1, enc1 = load_encoding('samples/truncated.mp3', sample_length)

fig, axs = plt.subplots(1)
axs.plot(enc1[0])
axs.set_title('Truncated "Sixties Honeybee Dialect" Encoding')

fastgen.synthesize(
    enc1,
    save_paths=['gen_honeybee.wav'],
    checkpoint_path="model/model.ckpt-200000",
    samples_per_save=sample_length)
