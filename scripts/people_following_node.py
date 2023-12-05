#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from simple_RobotGuidance.msg import ClusterInfo
from sklearn.cluster import DBSCAN

class PeopleFollowingNode:
    def __init__(self):
        rospy.init_node('people_following_node', anonymous=True)

        # パラメータ
        self.min_points_per_cluster = rospy.get_param("~min_points_per_cluster", 10)
        self.lidar_topic = rospy.get_param("~lidar_topic", "/scan")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.cluster_info_topic = rospy.get_param("~cluster_info_topic", "/cluster_info")

        # パブリッシャとサブスクライバ
        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.cluster_info_pub = rospy.Publisher(self.cluster_info_topic, ClusterInfo, queue_size=1)
        rospy.Subscriber(self.lidar_topic, LaserScan, self.lidar_callback)

    def lidar_callback(self, data):
        # レーザースキャンデータを取得
        ranges = np.array(data.ranges)

        # NaNを無視して2D座標に変換
        valid_indices = ~np.isnan(ranges)
        points = np.vstack((ranges[valid_indices], np.zeros(np.sum(valid_indices))))

        # DBSCANでクラスタリング
        db = DBSCAN(eps=0.1, min_samples=self.min_points_per_cluster).fit(points.T)

        # クラスタラベル取得
        labels = db.labels_

        # 人のクラスタ（ラベル0）を見つける
        people_cluster_indices = np.where(labels == 0)[0]

        if len(people_cluster_indices) > 0:
            # 人のクラスタの中心座標を計算
            people_cluster_center = np.mean(points[:, people_cluster_indices], axis=1)

            # 新しいメッセージを作成して発行
            cluster_msg = ClusterInfo()
            cluster_msg.cluster_x = points[0, people_cluster_indices].tolist()
            cluster_msg.cluster_y = points[1, people_cluster_indices].tolist()
            self.cluster_info_pub.publish(cluster_msg)

            # ロボットを人の方に向けるような速度ベクトルを生成
            cmd_vel_msg = Twist()
            cmd_vel_msg.angular.z = np.arctan2(people_cluster_center[1], people_cluster_center[0])
            cmd_vel_msg.linear.x = 0.2  # 前進速度

            # 速度をパブリッシュ
            self.cmd_vel_pub.publish(cmd_vel_msg)

if __name__ == '__main__':
    try:
        node = PeopleFollowingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass