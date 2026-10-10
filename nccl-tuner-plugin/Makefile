# NCCL tuner plugin 骨架
#
# 本机验证（用自带的 stub 头）：
#   make test        编译测试并跑
#   make so          编出 libnccl-tuner-mycard.so（真实用法）
#
# 挂到真 NCCL 上（需要 NCCL 源码里的真头文件）：
#   make so NCCL_INC=/path/to/nccl/src/include/plugin
#   export NCCL_TUNER_PLUGIN=$PWD/libnccl-tuner-mycard.so

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -Werror -O2 -fPIC
NCCL_INC ?= .

.PHONY: all test so clean

all: test so

test: tuner_test
	./tuner_test

tuner_test: tuner_test.cpp mycard_tuner.cpp nccl_tuner.h
	$(CXX) $(CXXFLAGS) -I$(NCCL_INC) -o $@ tuner_test.cpp mycard_tuner.cpp

# 真实形态：编成一个 .so，NCCL 用 dlopen 加载
so: libnccl-tuner-mycard.so

libnccl-tuner-mycard.so: mycard_tuner.cpp nccl_tuner.h
	$(CXX) $(CXXFLAGS) -I$(NCCL_INC) -shared -o $@ mycard_tuner.cpp

clean:
	rm -f tuner_test libnccl-tuner-mycard.so
