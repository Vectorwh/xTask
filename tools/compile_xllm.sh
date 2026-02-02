#!/bin/bash
# xllm 内网编译下载部署

git clone https://github.com/jd-opensource/xllm.git

chmod +x ./cibuild/build_npu.sh
bash cibuild/build_npu.sh 'pip install pre-commit -i https://pypi.tuna.tsinghua.edu.cn/simple; python setup.py bdist_wheel; pip install dist/* --force-reinstall;'

rm -rf /usr/local/bin/xllm
cp /usr/local/lib64/python3.11/site-packages/xllm/xllm /usr/local/bin/

echo "Installation completed successfully!"
