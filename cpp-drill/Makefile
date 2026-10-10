# C++ 手撕练习 —— 针对通信库 / 基础设施岗
#
#   make        编译全部 + 逐个跑测试
#   make clean  清理
#
# 严格模式：-Wall -Wextra -Werror（问答写代码也要零警告）

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -Werror -O2 -pthread

SRCS := $(wildcard [0-9]*.cpp)
BINS := $(patsubst %.cpp,%,$(SRCS))

.PHONY: all clean run

all: $(BINS)
	@echo
	@echo "============ 全部编译通过，逐个跑测试 ============"
	@for b in $(BINS); do ./$$b || { echo "!!! $$b 失败"; exit 1; }; echo; done
	@echo "************ 8 道全部通过 ************"

run: all

%: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(BINS)
