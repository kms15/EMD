EDFLIBDIR=./lib/edflib_111
CXXFLAGS=--std=c++11 -O2 -Wall -Wpedantic -Werror -I$(EDFLIBDIR)
LDFLAGS=-static -O2
SOURCES=EMD.cpp
LIBSOURCES=$(EDFLIBDIR)/edflib.c
OBJECTS=$(SOURCES:%.cpp=%.o) $(LIBSOURCES:%.c=%.o)
TARGET=EMD
FIGURES=SC4012E0-PSG-hilbert.png SC4012E0-PSG-spectrogram.png

$(TARGET): $(OBJECTS)
	$(CXX) -o $@ $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS)

clean:
	rm -f $(TARGET) $(OBJECTS) \
		$(TARGET)_covgen \
		$(OBJECTS:%.o=%_covgen.gcno) $(OBJECTS:%.o=%_covgen.gcda)\
		$(OBJECTS:%.o=%_covgen.o) $(SOURCES:%=%.gcov)\
		SC*.edf *-spectrum.csv *-hilbert.png *-spectrogram.png\
		edflib.gcno edflib.gcda

#############################################################################
# Code coverage
#############################################################################

%_covgen.o : %.cpp
	$(CXX) -o $@ $(CPPFLAGS) $(CXXFLAGS) \
		-O0 -fprofile-arcs -ftest-coverage -c $^

%.cpp.gcov : %.cpp $(TARGET)_covgen
	gcov $< -r -o $(<:%.cpp=%_covgen.o)

$(TARGET)_covgen: $(SOURCES:%.cpp=%_covgen.o) $(LIBSOURCES)
	$(CXX) -o $@ $(LDFLAGS) $(LOADLIBES) $(LDLIBS) \
		-O0 -fprofile-arcs -ftest-coverage $^
	! ./$(TARGET)_covgen
	./$(TARGET)_covgen --run-tests

coverage: $(SOURCES:%=%.gcov)

#############################################################################
# Plots and figures
#############################################################################

# files from The Sleep-EDF Database
SC%.edf :
	wget http://www.physionet.org/physiobank/database/sleep-edfx/$@

%-spectrum.csv : %.edf $(TARGET)
	./$(TARGET) $< --generate-spectrum $@

%-hilbert.png : %-spectrum.csv plot-hilbert-spectrum.py
	python plot-hilbert-spectrum.py $< -o $@

%-spectrogram.png : %.edf plot-spectrogram.py
	python plot-spectrogram.py $< -o $@

figures : $(FIGURES)
