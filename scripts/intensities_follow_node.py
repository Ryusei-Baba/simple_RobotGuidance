#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import csv
import os
import rospkg
from datetime import datetime

class FollowReflector:
    def __init__(self):
        rospy.init_node('follow_reflector', anonymous=True)

        # パッケージのパスを取得
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('simple_RobotGuidance')

        # パスを初期化
        self.path = os.path.join(package_path, 'data/intensities')

        # フォルダが存在しなければ作成
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # 現在の日時を取得してファイル名に含める
        current_time = datetime.now().strftime("%Y%m%d_%H:%M:%S")
        self.csv_file_path = os.path.join(self.path, f'lidar_data_{current_time}.csv')

        # CSVファイルに書き込むためのファイルオブジェクトを作成
        self.csv_file = open(self.csv_file_path, 'a')
        self.csv_writer = csv.writer(self.csv_file, lineterminator='\n')
        self.csv_writer.writerow(['Timestamp', 'max_intensity_index', 'len_reflectance_values', 'cmd_vel.angular.z'])

        rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.vel = Twist()
        self.vel.linear.x = 0.0
        self.vel.angular.z = 0.0

    def laser_callback(self, laser_data):
        # レーザースキャンデータを処理（LiDARデータの総数）
        reflectance_values = laser_data.intensities

        # 最も反射強度の高い方向（リスト内の最大値）を見つける
        max_intensity_index = reflectance_values.index(max(reflectance_values))

        # 速度コマンドを生成
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.2  # 前進速度
        cmd_vel.angular.z = self.calculate_angular_velocity(max_intensity_index, len(reflectance_values))
        # print(cmd_vel.angular.z)

        # 速度コマンドをパブリッシュ
        self.cmd_vel_pub.publish(cmd_vel)

        # CSVファイルにデータを書き込む
        self.csv_writer.writerow([rospy.get_time(), max_intensity_index, len(reflectance_values), cmd_vel.angular.z])

    def calculate_angular_velocity(self, max_intensity_index, num_points):  # num_points:1回のスキャンで採集したデータの総数（検出範囲360度で1度ごとにデータが取得されている場合、num_pointsは360）
        # 角速度の計算（要調整）
        max_intensity_angle = max_intensity_index / float(num_points) * 2.0 * 3.141592653589793 - 3.141592653589793
        # max_intensity_indexをnum_pointsで割り、0から1の範囲に正規化
        # 結果を2倍し、0から2の範囲に拡大
        # 最後にπを掛け、角度に変換
        # 3.141592653589793を引くことで、範囲を-πからπに変換

        # 角速度を範囲[-0.5, 0.5]に正規化
        normalized_angular_velocity = max_intensity_angle / 3.141592653589793 * 0.5
        return normalized_angular_velocity

if __name__ == '__main__':
    try:
        follower = FollowReflector()
        rospy.spin()
    except rospy.ROSInterruptException:
        # プログラムが終了するときにCSVファイルを閉じる
        if follower.csv_file:
            follower.csv_file.close()
