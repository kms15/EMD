#!/usr/bin/python
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg');
# use a non-interactive renderer
import matplotlib.pyplot as plt
import edflib

if len(sys.argv) != 4 or sys.argv[2] != "-o":
    print("usage:\n\t{0} datafile.edf -o plotfile.png\n".format(sys.argv[0]));
else:

    # load the data
    signal_num = 0 # TODO: allow this to be specified on the command line!
    edf = edflib.EdfReader(sys.argv[1])
    data = edf.readSignal(signal_num)
    x_bin_size = 8192

    # plot the data
    plt.figure(1, figsize=(8, 3))
    plt.specgram(data, Fs=edf.samplefrequency(signal_num),
            NFFT=x_bin_size, noverlap=x_bin_size/2)
    plt.xlim(0, data.shape[0]/edf.samplefrequency(signal_num))
    plt.ylim(0, edf.samplefrequency(signal_num)/2)
    plt.title("Spectrogram of {0}".format(sys.argv[1]));
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(sys.argv[3], dpi=300)

