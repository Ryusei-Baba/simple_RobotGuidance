#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
import csv
import os
import rospkg
from datetime import datetime

class IntensitiesData:
    def __init__(self):
        rospy.init_node('intensities_data', anonymous=True)

        # intensitiesを取得した回数をカウントする変数
        self.intensities_count = 0  

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
        self.csv_file_path = os.path.join(self.path, f'intensities_data_{current_time}.csv')

        # CSVファイルに書き込むためのファイルオブジェクトを作成
        self.csv_file = open(self.csv_file_path, 'a')
        self.csv_writer = csv.writer(self.csv_file, lineterminator='\n')
        self.csv_writer.writerow(['Timestamp', 'intensities'])

        rospy.Subscriber('/scan', LaserScan, self.laser_callback)

    def laser_callback(self, data):
        timestamp = rospy.get_time()
        intensities = data.intensities

        # CSVファイルにデータを書き込む
        self.csv_writer.writerow([rospy.get_time(), intensities])

        # intensitiesを取得した回数を更新
        self.intensities_count += 1

        # intensitiesを取得した回数を出力
        rospy.loginfo(f"count: {self.intensities_count}, value: {intensities}")

if __name__ == '__main__':
    try:
        follower = IntensitiesData()
        rospy.spin()
    except rospy.ROSInterruptException:
        # プログラムが終了するときにCSVファイルを閉じる
        if follower.csv_file:
            follower.csv_file.close()
