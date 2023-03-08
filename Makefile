CC=g++
CFLAGS=-c -Wall -mavx
SOURCES=analyze_query.cpp
OBJECTS=$(SOURCES:.cpp=.o)
INCLUDES= -I/home/tianbin/smartann/HKmeans/util
EXECUTABLE=analyze_query

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	@mkdir -p output/
	$(CC) $(LDFLAGS) $(OBJECTS) -o output/$(EXECUTABLE)

%.o: %.cpp
	$(CC) $(INCLUDES) $(CFLAGS) $< -o $@ -w

clean:
	rm -rf $(OBJECTS) output/
