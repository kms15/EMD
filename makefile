EDFLIBDIR=./lib/edflib_111
CXXFLAGS=--std=c++11 -O2 -Wall -Wpedantic -Werror -I$(EDFLIBDIR)
LDFLAGS=-static -O2
HEADERS=emd.h
SOURCES=main.cpp
LIBSOURCES=$(EDFLIBDIR)/edflib.c
OBJECTS=$(SOURCES:%.cpp=%.o) $(LIBSOURCES:%.c=%.o)
TARGET=emd
FIGURES=SC4012E0-PSG-hilbert.png SC4012E0-PSG-spectrogram.png

$(TARGET): $(OBJECTS)
	$(CXX) -o $@ $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS)

clean:
	rm -f $(TARGET) $(OBJECTS) \
		$(TARGET)_covgen \
		$(OBJECTS:%.o=%_covgen.gcno) $(OBJECTS:%.o=%_covgen.gcda)\
		$(OBJECTS:%.o=%_covgen.o) $(SOURCES:%=%.gcov)\
		$(HEADERS:%=%.gcov)\
		SC*.edf *-spectrum.csv *-hilbert.png *-spectrogram.png\
		edflib.gcno edflib.gcda \
		bdf_test_generator test_generator.bdf test_spectrum.csa\
		$(TARGET)_coverage_tests_done

#############################################################################
# Code coverage
#############################################################################

%_covgen.o : %.cpp
	$(CXX) -o $@ $(CPPFLAGS) $(CXXFLAGS) \
		-O0 -fprofile-arcs -ftest-coverage -c $^

%.cpp.gcov : %.cpp $(TARGET)_coverage_tests_done
	gcov $< -r -o $(<:%.cpp=%_covgen.o)

# runs both unit tests and integration tests to get full coverage
$(TARGET)_coverage_tests_done : $(TARGET)_covgen test_generator.bdf
	./$(TARGET)_covgen test_generator.bdf \
		--generate-spectrum test_spectrum.csv
	! ./$(TARGET)_covgen # usage
	! ./$(TARGET)_covgen non-existant-file \
		--generate-spectrum dummy-spectrum
	! ./$(TARGET)_covgen --generate-spectrum dummy-spectrum
	! ./$(TARGET)_covgen test_generator.bdf \
		--generate-spectrum dummy-spectrum1 \
		--generate-spectrum dummy-spectrum2
	! ./$(TARGET)_covgen test_generator.bdf --generate-spectrum
	! ./$(TARGET)_covgen test_generator.bdf \
		--generate-spectrum /non-existant-dir/dummy-spectrum
	! ./$(TARGET)_covgen test_generator.bdf test_generator.bdf \
		--generate-spectrum test_spectrum.csv
	! ./$(TARGET)_covgen test_generator.bdf --undefined-option \
		--generate-spectrum test_spectrum.csv
	./$(TARGET)_covgen --run-tests
	touch $@

$(TARGET)_covgen: $(SOURCES:%.cpp=%_covgen.o) $(LIBSOURCES)
	$(CXX) -o $@ $(LDFLAGS) $(LOADLIBES) $(LDLIBS) \
		-O0 -fprofile-arcs -ftest-coverage $^

coverage: $(SOURCES:%=%.gcov)

bdf_test_generator : $(EDFLIBDIR)/test_generator.c $(EDFLIBDIR)/edflib.c
	$(CXX) -o $@ $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) -I$(EDFLIBDIR)

test_generator.bdf : bdf_test_generator
	./$<

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
