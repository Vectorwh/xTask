#!/bin/bash
# xllm 内网编译下载部署

cd /export/home/kongshuaikang1/
git clone https://gitee.com/mirrors/xllm.git
git config --global --add safe.directory /export/home/kongshuaikang1/xllm
git config --global --add safe.directory /export/home/kongshuaikang1/xllm/third_party/xllm_ops
git config --global --add safe.directory /export/home/kongshuaikang1/xllm/third_party/brpc
git config --global --add safe.directory /export/home/kongshuaikang1/xllm/third_party/etcd_cpp_apiv3
git config --global --add safe.directory /export/home/kongshuaikang1/xllm/third_party/minja
git config --global --add safe.directory /export/home/kongshuaikang1/xllm/third_party/smhasher
git config --global --add safe.directory /export/home/kongshuaikang1/xllm/third_party/torch_npu_ops
git config --global --add safe.directory /export/home/kongshuaikang1/xllm/third_party/Mooncake
git config --global --add safe.directory /export/home/kongshuaikang1/xllm/third_party/cpprestsdk
git config --global --add safe.directory /export/home/kongshuaikang1/xllm/third_party/hccl_transfer
git config --global --add safe.directory /export/home/kongshuaikang1/xllm/third_party/sentencepiece
git config --global --add safe.directory /export/home/kongshuaikang1/xllm/third_party/spdlog

cd /export/home/kongshuaikang1/xllm/
git pull
git submodule init
git submodule update

chmod +x ./cibuild/build_npu.sh
bash ./cibuild/build_npu.sh 'pip install pre-commit -i https://pypi.tuna.tsinghua.edu.cn/simple; python setup.py bdist_wheel; pip install dist/* --force-reinstall;'

echo "Installation completed successfully!"
