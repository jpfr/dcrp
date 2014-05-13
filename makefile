TARGET = dcrp
LIBS = -lm -larmadillo
CPP = g++
CFLAGS = -O0 -g -Wall -std=c++11 -pedantic -pipe -fPIC -fno-exceptions -fstack-protector -Wl,-z,relro -Wl,-z,now -fvisibility=hidden -W -Wall -Wno-unused-parameter -Wno-unused-function -Wno-unused-label -Wpointer-arith -Wformat -Wreturn-type -Wsign-compare -Wmultichar -Wformat-nonliteral -Winit-self -Wuninitialized -Wno-deprecated -Wformat-security -Werror -fexceptions

.PHONY: clean all default

default: $(TARGET)
all: default

OBJECTS = $(patsubst %.cpp, %.o, $(wildcard *.cpp))
HEADERS = $(wildcard *.h)

%.o: %.cpp $(HEADERS)
	$(CPP) $(CFLAGS) -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CPP) $(OBJECTS) -Wall $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TARGET)
