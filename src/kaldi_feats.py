from kaldi.matrix import SubVector #, SubMatrix
from kaldi.util.options import ParseOptions
import numpy as np
from config import LOG_MEL_FILTERS
from config import VAD_WIDTH_MS, VAD_SHIFT_MS, LOG_MEL_SHIFT_MS, LOG_MEL_WIDTH_MS
import sys

from kaldi.feat.window import FrameExtractionOptions

"""logmel stuff"""
from kaldi.feat.fbank import Fbank, FbankOptions

"""plp stuff """
from kaldi.feat.plp import Plp, PlpOptions

def logmel_feats(*, signal, sampling_rate):
    if signal.dtype == np.int16:
        logmel_opts = FbankOptions()

        logmel_opts.frame_opts.frame_length_ms=LOG_MEL_WIDTH_MS
        logmel_opts.frame_opts.frame_shift_ms=LOG_MEL_SHIFT_MS 
        logmel_opts.frame_opts.samp_freq = sampling_rate

        logmel_opts.mel_opts.num_bins = LOG_MEL_FILTERS
        fbank = Fbank(logmel_opts)

        sf = logmel_opts.frame_opts.samp_freq
        channel = SubVector(signal)
        feats = fbank.compute_features(channel, sf, 1.0)
        return np.array(feats)
    else: 
        sys.exit("PYKALDI EXPECT np.int16 dtype input")



def plp_feats(*, signal, sampling_rate):
    if signal.dtype == np.int16:
        plp_opts = PlpOptions()
        plp_opts.frame_opts.frame_length_ms=VAD_WIDTH_MS
        plp_opts.frame_opts.frame_shift_ms=VAD_SHIFT_MS 
        plp_opts.frame_opts.samp_freq = sampling_rate

        plp = Plp(plp_opts)
        sf = plp_opts.frame_opts.samp_freq 
        channel = SubVector(signal) 
        feats = plp.compute_features(channel, sf, 1.0)

        return np.array(feats)
    else:
        sys.exit("PYKALDI EXPECT np.int16 dtype input")

    """mono"""
    # m = SubVector(np.mean(s, axis=0))
