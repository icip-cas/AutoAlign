docker run -d --name funny_sammet --privileged --user root \
            --shm-size 500g \
            -e HCCL_IF_BASE_PORT=50000 \
            --device=/dev/davinci0 \
            --device=/dev/davinci1 \
            --device=/dev/davinci2 \
            --device=/dev/davinci3 \
            --device=/dev/davinci4 \
            --device=/dev/davinci5 \
            --device=/dev/davinci6 \
            --device=/dev/davinci7 \
            --device=/dev/davinci_manager \
            --device=/dev/hisi_hdc \
            --device /dev/devmm_svm \
            -v /usr/local/dcmi:/usr/local/dcmi \
            -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
            -v /usr/local/sbin:/usr/local/sbin \
            -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
            -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
            -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
            -v /etc/hccn.conf:/etc/hccn.conf \
            -v $(pwd):/home/ma-user/AutoAlign \
            -v /mnt/sfs_turbo/hf_models:/home/ma-user/hf_models \
            autoalign-dev:latest \
            tail -f /dev/null