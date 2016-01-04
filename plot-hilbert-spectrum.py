#!/usr/bin/python
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg');
# use a non-interactive renderer
import matplotlib.pyplot as plt

if len(sys.argv) != 4 or sys.argv[2] != "-o":
    print("usage:\n\t{0} spectrumfile.csv -o plotfile.png\n".format(sys.argv[0]));
else:

    # load and plot
    data = np.loadtxt(sys.argv[1], delimiter=",").transpose()
    # TODO: read these constants from the file!
    x_bin_size = 8192
    sample_frequency = 100.
    fraction_frequencies_shown = 3./4

    # plot the data
    plt.figure(1, figsize=(8, 3))

    plt.imshow(
            np.log(data[1:int(data.shape[0]*fraction_frequencies_shown),:]),
            aspect='auto', origin='lower',
            extent=[0, data.shape[1] * x_bin_size / sample_frequency,
                sample_frequency/int(data.shape[0]), sample_frequency/2*fraction_frequencies_shown
                ]
            )
    plt.title("Hilbert spectrum of {0}".format(sys.argv[1]));
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(sys.argv[3], dpi=300)

