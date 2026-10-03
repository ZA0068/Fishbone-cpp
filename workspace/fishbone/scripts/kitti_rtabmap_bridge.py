#!/usr/bin/python3
# NOTE: pinned to the system interpreter, not `env python3` -- PATH resolves
# python3 to conda's copy first in this environment, and rclpy's compiled
# bindings are built against the system python3.12, not conda's python3.11.
"""Bridges our custom fishbone::msg::StereoCameraData into standard
sensor_msgs/Image + CameraInfo topics that rtabmap's stereo_odometry expects.
Also publishes a TimeReference per frame (stamp + KITTI frame_id in `source`)
so a downstream recorder can recover the true frame_id by matching stamps,
since rtabmap's odometry output echoes the input image's stamp exactly."""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, TimeReference
from fishbone.msg import StereoCameraData


class KittiBridge(Node):
    def __init__(self):
        super().__init__('kitti_rtabmap_bridge')

        # From calib.txt P2 (left) / P3 (right) -- fx=fy=718.856, cx=607.19, cy=185.22
        fx, fy, cx, cy = 718.856, 718.856, 607.1928, 185.2157
        self.left_info = self._make_camera_info(fx, fy, cx, cy, tx=45.38225)
        self.right_info = self._make_camera_info(fx, fy, cx, cy, tx=-337.2877)

        self.left_img_pub = self.create_publisher(Image, '/kitti/left/image_rect', 10)
        self.right_img_pub = self.create_publisher(Image, '/kitti/right/image_rect', 10)
        self.left_info_pub = self.create_publisher(CameraInfo, '/kitti/left/camera_info', 10)
        self.right_info_pub = self.create_publisher(CameraInfo, '/kitti/right/camera_info', 10)
        self.frame_ref_pub = self.create_publisher(TimeReference, '/kitti/frame_ref', 20)

        self.sub = self.create_subscription(StereoCameraData, '/fishbone/stereo_camera', self.cb, 10)
        self.get_logger().info('KITTI->rtabmap bridge started')

    def _make_camera_info(self, fx, fy, cx, cy, tx):
        info = CameraInfo()
        info.width = 1241
        info.height = 376
        info.distortion_model = 'plumb_bob'
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, tx, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def cb(self, msg: StereoCameraData):
        stamp = msg.left_camera.image.header.stamp
        fid = int(msg.left_camera.frame_id)
        left_img = msg.left_camera.image
        right_img = msg.right_camera.image
        left_img.header.frame_id = 'camera_left'
        right_img.header.frame_id = 'camera_right'
        self.left_img_pub.publish(left_img)
        self.right_img_pub.publish(right_img)
        self.left_info.header.stamp = stamp
        self.left_info.header.frame_id = 'camera_left'
        self.right_info.header.stamp = stamp
        self.right_info.header.frame_id = 'camera_right'
        self.left_info_pub.publish(self.left_info)
        self.right_info_pub.publish(self.right_info)

        ref = TimeReference()
        ref.header.stamp = stamp
        ref.source = str(fid)
        self.frame_ref_pub.publish(ref)


def main():
    rclpy.init()
    node = KittiBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
