CPP = g++ -std=c++11
BIN = ./bin/
RM 	= rm -rf
SRC	= ./SBPNN/
objects = main.o CBPNet.o
all: $(objects)
	mkdir -p $(BIN)
	$(CPP) -o $(BIN)net $(objects)
main.o: $(SRC)cbpnn_main.cpp $(SRC)CBPNet.hpp
	$(CPP) -c -o main.o $(SRC)cbpnn_main.cpp
CBPNet.o:$(SRC)CBPNet.cpp
	$(CPP) -c $(SRC)CBPNet.cpp
clean:
	$(RM) $(objects) $(BIN)
	
