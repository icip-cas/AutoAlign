docker run -d --name autoalign_a100 \
           --gpus all \
           --ipc=host \
           --shm-size 200g \
           -w /opt/AutoAlign \
           -e SWANLAB_API_KEY=${SWANLAB_API_KEY} \
           -v $(pwd):/opt/AutoAlign \
           -v /ceph_home:/ceph_home \
           autoalign-megatron-nvidia:dev \
           tail -f /dev/null
