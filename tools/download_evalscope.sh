#!/bin/bash
# evalscope编译与部署
cd /workspace/benchmark/
pip3 install evalscope 
if [ $? -ne 0 ]; then
    echo "Error: Failed to install evalscope"
    exit 1
fi

pip3 install 'evalscope[perf]'
if [ $? -ne 0 ]; then
    echo "Error: Failed to install evalscope[perf]"
    exit 1
fi


cd /workspace
echo "Installation evalscope successfully!"
