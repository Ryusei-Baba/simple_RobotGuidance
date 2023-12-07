#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage  # 画像を出力するために追加

class FollowerNode:
    def __init__(self):
        rospy.init_node('follower_node', anonymous=True)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera_center/usb_cam/image_raw', Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # 画像を出力するための設定
        self.image_pub = rospy.Publisher('/output/image_raw/compressed', CompressedImage, queue_size=1)

        self.target_color_lower = np.array([100, 0, 0], dtype=np.uint8)
        self.target_color_upper = np.array([255, 100, 100], dtype=np.uint8)

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except Exception as e:
            print(e)
            return

        # 画像処理: カラーフィルタリング
        mask = cv2.inRange(cv_image, self.target_color_lower, self.target_color_upper)
        result = cv2.bitwise_and(cv_image, cv_image, mask=mask)

        # 画像処理: 輪郭検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 最大の輪郭を取得
            max_contour = max(contours, key=cv2.contourArea)

            # 輪郭の中心座標を計算
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # ロボットの制御コマンド生成
                twist_msg = Twist()
                twist_msg.linear.x = 0.2  # 速度調整
                twist_msg.angular.z = 0.002 * (cx - cv_image.shape[1] / 2)  # 中心からのずれに比例して回転

                # 画像に検出範囲とangular.zを描画
                cv2.drawContours(cv_image, [max_contour], -1, (0, 255, 0), 2)  # 検出範囲を描画
                cv2.circle(cv_image, (cx, cy), 5, (0, 0, 255), -1)  # 中心を描画
                cv2.putText(cv_image, f"Angular Z: {twist_msg.angular.z:.4f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # angular.zを描画

                # 画像を出力
                compressed_image_msg = CompressedImage()
                compressed_image_msg.header.stamp = rospy.Time.now()
                compressed_image_msg.format = "jpeg"
                _, compressed_image_msg.data = cv2.imencode('.jpg', cv_image)
                self.image_pub.publish(compressed_image_msg)

                # ロボットの制御コマンドをパブリッシュ
                self.cmd_vel_pub.publish(twist_msg)

        # ウィンドウに生の画像を表示
        cv2.imshow("Raw Image", cv_image)
        cv2.waitKey(1)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    follower_node = FollowerNode()
    follower_node.run()
