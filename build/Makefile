PROG  = test
CC    = g++
FLAGS = -Wall -I..\include
SRCS  = ..\main.cpp ..\src\matrix.cpp ..\src\nn.cpp
OBJS  = main.o matrix.o nn.o

$(PROG):
	$(CC) -c $(SRCS) $(FLAGS)
	$(CC) $(OBJS) $(FLAGS) -o $(PROG)

clean:
	del $(PROG).exe $(OBJS)
