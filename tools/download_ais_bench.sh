#!/bin/bash
# ais_bench编译与部署
#cd /workspace
pip uninstall ais-bench-benchmark -y
#git clone https://github.com/AISBench/benchmark.git
if [ $? -ne 0 ]; then
    echo "Error: Failed to uninstall ais_bench"
    exit 1
fi

cp -r /xllm_test_060/benchmark /workspace/

cd /workspace/benchmark/
pip3 install -e ./ --use-pep517
if [ $? -ne 0 ]; then
    echo "Error: Failed to install ais_bench Python package"
    exit 1
fi

cd /workspace
echo "Installation ais_bench successfully!"