library(edf);
library(EMD);
library(Rlibeemd);
print("loading file...");
print(system.time(d <- read.edf("~/Downloads/SC4012E0-PSG.edf")));
print("performing emd...");
print(system.time(d.emd <- emd(d$signal$EEG_Fpz_Cz$data)));
print("finding Hilbert spectrum...");
print(system.time(d.hilb <- hilbertspec(d.emd)));


