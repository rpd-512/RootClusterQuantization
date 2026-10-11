# ---- toolchain -------------------------------------------------------------------------------
CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
# `make OMP=` builds the RCQ targets single-threaded
OMP      ?= -fopenmp
INCLUDES := -Iincludes -I/usr/include/eigen3
BUILD    := build_make
DEPFLAGS := -MMD -MP

# ---- targets ---------------------------------------------------------------------------------
# Eigen demo (main.cpp + example generator)
DEMO     := RCQunatSearch
# original rootfit tests
TEST     := test_rootfit
# exact RCQ driver
RCQ_CLI  := rcq_cli
# exact RCQ tests
RCQ_TEST := test_rcq

DEMO_SRCS    := main.cpp includes/example_generator.cpp
TEST_SRCS    := includes/test_rootfit.cpp
RCQ_CLI_SRCS := includes/rcq_cli.cpp
RCQ_TEST_SRCS:= includes/test_rcq.cpp

ALL_SRCS := $(DEMO_SRCS) $(TEST_SRCS) $(RCQ_CLI_SRCS) $(RCQ_TEST_SRCS)
OBJ      = $(patsubst %.cpp,$(BUILD)/%.o,$(1))

.PHONY: all run test rcq test-rcq check clean help
.DEFAULT_GOAL := all

all: $(DEMO) $(TEST) $(RCQ_CLI) $(RCQ_TEST)

$(DEMO):     $(call OBJ,$(DEMO_SRCS))
$(TEST):     $(call OBJ,$(TEST_SRCS))
$(RCQ_CLI):  $(call OBJ,$(RCQ_CLI_SRCS))
$(RCQ_TEST): $(call OBJ,$(RCQ_TEST_SRCS))

$(DEMO) $(TEST) $(RCQ_CLI) $(RCQ_TEST):
	$(CXX) $(CXXFLAGS) $^ -o $@

# OpenMP only for the RCQ targets (the pragmas are guarded; without OMP the code runs serially)
$(RCQ_CLI) $(RCQ_TEST) $(call OBJ,$(RCQ_CLI_SRCS) $(RCQ_TEST_SRCS)): CXXFLAGS += $(OMP)
$(RCQ_CLI) $(call OBJ,$(RCQ_CLI_SRCS)): CXXFLAGS += -O3

# One rule for every object; headers are tracked automatically through the generated .d files.
$(BUILD)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEPFLAGS) -c $< -o $@

-include $(patsubst %.cpp,$(BUILD)/%.d,$(ALL_SRCS))

# ---- convenience -----------------------------------------------------------------------------
run: $(DEMO)
	./$(DEMO)

test: $(TEST)
	./$(TEST)

rcq: $(RCQ_CLI)                           # usage: ./rcq_cli 8 6 --system random
	@echo "built ./$(RCQ_CLI)   e.g.  ./$(RCQ_CLI) 8 all --system random"

test-rcq: $(RCQ_TEST)
	./$(RCQ_TEST)

check: test test-rcq                      # all test suites

clean:
	rm -rf $(BUILD) $(DEMO) $(TEST) $(RCQ_CLI) $(RCQ_TEST) demo

help:
	@echo "targets: all run test rcq test-rcq check clean"
	@echo "vars:    CXX CXXFLAGS OMP (empty = serial RCQ) BUILD"