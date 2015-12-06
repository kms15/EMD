CPPFLAGS=--std=c++11 -O2 -Wall -Wpedantic -Werror
LDFLAGS=-static -O2
OBJECTS=EMD.o
TARGET=EMD

$(TARGET): $(OBJECTS)
	$(CXX) -o $@ $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS)

clean:
	rm -f $(TARGET) $(OBJECTS)
