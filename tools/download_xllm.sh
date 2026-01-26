#!/bin/bash
# xllm 内网已编译的版本包下载部署

rm -rf xllm_pack xllm_pack.tar.gz
#wget https://9n-das-tools.s3-internal.cn-north-1.jdcloud-oss.com/xllm-ai/xllm_release/0.7.0/a3-arm/xllm_pack.tar.gz
# A3
wget https://9n-das-tools.s3.cn-north-1.jdcloud-oss.com/xllm-ai/xllm_release/0.8.0/a3-arm/xllm_pack.tar.gz
# A2
# wget https://9n-das-tools.s3.cn-north-1.jdcloud-oss.com/xllm-ai/xllm_release/0.7.0/a2-x86/xllm_pack.tar.gz

tar -zxvf xllm_pack.tar.gz
if [ $? -ne 0 ]; then
    echo "Error: Failed to extract xllm_pack.tar.gz"
    exit 1
fi

cd xllm_pack || {
    echo "Error: Failed to enter xllm_pack directory"
    exit 1
}

chmod 777 etcd xllm_master_serving
yes | mv etcd xllm_master_serving /usr/local/bin/

tar -zxvf vendors-*.tar.gz
if [ $? -ne 0 ]; then
    echo "Error: Failed to extract vendors-*.tar.gz"
    exit 1
fi

for file in vendors/*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        rm -f "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/$filename"
        mv "$file" "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/"
    fi
done

tar -zxvf xllm-lib-*.tar.gz
if [ $? -ne 0 ]; then
    echo "Error: Failed to extract xllm-lib-*.tar.gz"
    exit 1
fi

yes | mv xllm-lib/* /usr/lib64/

chmod 777 xllm-*-linux*.whl
pip install xllm-*-linux*.whl
if [ $? -ne 0 ]; then
    echo "Error: Failed to install xllm Python package"
    exit 1
fi

rm -rf xllm-lib vendors xllm-lib-*.tar.gz vendors-*.tar.gz xllm-*-linux*.whl

rm -rf /usr/local/bin/xllm
cp /usr/local/lib64/python3.11/site-packages/xllm/xllm /usr/local/bin/

echo "Installation completed successfully!"