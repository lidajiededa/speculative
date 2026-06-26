# 1. 先停掉当前 vLLM 服务
pkill -f "vllm|api_server|EngineCore" || true
ray stop -f || true

# 2. 进入镜像里的 vllm-ascend 源码目录
cd /vllm-workspace/vllm-ascend

# 3. 确认你改的是这里的源码
git diff -- csrc/attention/recurrent_gated_delta_rule

# 4. 准备环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && source /usr/local/Ascend/nnal/atb/set_env.sh

git submodule update --init --recursive

# 如果容器里 npu-smi 可用，一般不用手动设 SOC_VERSION
# 如果 npu-smi 不可用，再按机器设：
# Atlas A2/910B: export SOC_VERSION=ascend910b1
# Atlas A3:      export SOC_VERSION=ascend910_9391

export COMPILE_CUSTOM_KERNELS=1
export MAX_JOBS=$(nproc)

# 5. 清理旧编译产物，避免复用旧 so
rm -rf build dist *.egg-info
rm -f vllm_ascend/*.so
rm -rf vllm_ascend/_cann_ops_custom

# 6. 重新编译并安装
python -m pip install --no-build-isolation --no-deps -v -e .
