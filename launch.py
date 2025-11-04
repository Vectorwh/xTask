#!/usr/bin/env python3

docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:2.1.RC2-800I-A2-py311-openeuler24.03-lts

sudo docker run -dit --ipc=host -u 0 --name=mindie_2.1_j \
--privileged=true --shm-size=500g \
--privileged \
--network=host \
--ipc=host \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-v /export/home/weinan5:/export/home/weinan5 \
-w /export/home/weinan5/ \
-e https_proxy=http://bamboo-proxy.jd.com:80 \
-e https_proxy=http://bamboo-proxy.jd.com:80 \
swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:2.1.RC2-800I-A2-py311-openeuler24.03-lts /bin/bash