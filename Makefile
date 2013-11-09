CC=g++
CFLAGS=-c -Wall -Weffc++ -pedantic
# Note that it is CRITICAL that LDFLAGS be at the END of the compile directive
LDFLAGS=`pkg-config opencv --cflags --libs`

all: trainer performanceTest faceDetection

faceDetection: main.o images.o train.o detect.o performanceTest.o countImages.o lid.o
	$(CC) $^ -o $@ $(LDFLAGS)

# training is just the executable with TRAINING defined
trainer: mainTrainer.o images.o train.o detect.o performanceTest.o countImages.o lid.o
	$(CC) $^ -o $@ $(LDFLAGS)

# performanceTest is just the executable with PERFORMANCE_TEST defined
performanceTest: mainPerformanceTest.o images.o train.o detect.o performanceTest.o countImages.o lid.o
	$(CC) $^ -o $@ $(LDFLAGS)

# mainTrainer.o is just for training
mainTrainer.o: main.cpp images.hpp detect.hpp train.hpp performanceTest.hpp
	$(CC) -DTRAINING $(CFLAGS) -o $@ $<

# mainPerformanceTest.o is just for performance-testing
mainPerformanceTest.o: main.cpp images.hpp detect.hpp train.hpp performanceTest.hpp
	$(CC) -DPERFORMANCE_TEST $(CFLAGS) -o $@ $<

main.o: main.cpp images.hpp detect.hpp train.hpp performanceTest.hpp
	$(CC) $(CFLAGS) -o $@ $<

detect.o: detect.cpp detect.hpp images.hpp params.hpp
	$(CC) $(CFLAGS) -o $@ $<

train.o: train.cpp train.hpp params.hpp concatenate.hpp countImages.hpp
	$(CC) $(CFLAGS) -o $@ $<

images.o: images.cpp images.hpp
	$(CC) $(CFLAGS) -o $@ $<

coutImages.o: countImages.cpp countImages.hpp concatenate.hpp
	$(CC) $(CFLAGS) -o $@ $<

performanceTest.o: performanceTest.cpp performanceTest.hpp countImages.hpp params.hpp
	$(CC) $(CFLAGS) -o $@ $<

lid.o: lid.cpp lid.hpp params.hpp
	$(CC) $(CFLAGS) -o $@ $<


report: report.pdf purge

report.pdf: report.tex references.bib
	pdflatex report.tex


# Remove the stuff that pdflatex created
.PHONY: purge
purge:
	-rm report.aux report.log

.PHONY: clean
clean:
	-rm *.o trainer performanceTest faceDetection report.pdf


# General makefile notes:
#       - $@ is the executable name
#       - $< is the name of the first dependency
#       - $^ all dependencies
#       - Commands prefixed with "-" will ignore errors
