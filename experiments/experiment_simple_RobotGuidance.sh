for i in `seq 1`
do
  roslaunch simple_RobotGuidance simple_RobotGuidance.launch script:=simple_RobotGuidance_node.py
  sleep 10
done