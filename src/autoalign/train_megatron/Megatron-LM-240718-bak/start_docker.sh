docker run -it --gpus all --ipc=host --name mg \
    -v /data1:/data1 \
    dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch:24.07 \
    /bin/bash