# RDMA Verbs 最小示例 —— 依赖 libibverbs
#
#   Ubuntu/Debian:  sudo apt install -y rdma-core libibverbs-dev rdma-core-devel
#   CentOS/RHEL:    sudo yum install -y libibverbs-devel
#
CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -g
LDLIBS   ?= -libverbs

BINS := send_recv write_read

all: $(BINS)

# 每个示例 + 公共实现
%: %.cpp common.cpp common.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< common.cpp $(LDLIBS)

clean:
	rm -f $(BINS)

.PHONY: all clean
