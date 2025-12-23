import numpy as np
from constants import *


def stitch_wave_chunks(waves):
    """ 
    Takes wave chunks as a list and stitches them with cross-fades.
    """
    fade_len = WINDOW_SAMPLE_LEN - SIGNAL_HOP
    stitched_wave = np.array([])
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)

    for i, wave in enumerate(waves):
        if i > 0:
            # fade in wave chunks after the 1st one
            wave[:fade_len] *= fade_in

        if len(stitched_wave) > 0:
            stitched_wave[-fade_len:] += wave[:fade_len]
            stitched_wave = np.append(
                stitched_wave, wave[fade_len:])
        else:
            stitched_wave = np.append(stitched_wave, wave)

        # fade out all wave chunks
        stitched_wave[-fade_len:] *= fade_out

    stitched_wave /= (np.max(np.abs(stitched_wave)) + 0.02)
    return stitched_wave
