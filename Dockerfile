FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive

# ROS cv_bridge, CycloneDDS RMW, tf2_ros (for static TF so RViz2 Camera can display image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-cv-bridge \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-tf2-ros \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python deps for wall segmentation (no CUDA in base image; host nvidia runtime will provide GPU)
# Pin numpy<2 to be compatible with ROS cv_bridge (compiled against numpy 1.x)
RUN pip3 install --no-cache-dir \
    "numpy<2" \
    torch \
    torchvision \
    opencv-python \
    pyrealsense2 \
    tqdm \
    pillow

WORKDIR /app

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "run_realsense_ros2.py"]
