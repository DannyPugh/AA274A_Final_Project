#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Twist, Pose2D, PoseStamped
from std_msgs.msg import String
import tf
import numpy as np
from numpy import linalg
from utils.utils import wrapToPi
from utils.grids import StochOccupancyGrid2D
from planners import AStar, compute_smoothed_traj
import scipy.interpolate
import matplotlib.pyplot as plt
from controllers import PoseController, TrajectoryTracker, HeadingController
from enum import Enum

from dynamic_reconfigure.server import Server
from asl_turtlebot.cfg import NavigatorConfig

#HERE need to import new msg type
#need to recognize DetectedObject from detector.py
from asl_turtlebot.msg import DetectedObject
#STOP: do not need new type use Pose 2D from asl_turtlebot.msg import StopSign

#MY CHANGES for Markers
from visualization_msgs.msg import Marker

#RECOVER import data type for lidar scans
from sensor_msgs.msg import LaserScan

# state machine modes, not all implemented
class Mode(Enum):
    IDLE = 0
    ALIGN = 1
    TRACK = 2
    PARK = 3
    PAUSE = 4 #added
    PARK_ALIGN = 5 #added
    RECOVER = 6 #added

class Navigator:
    """
    This node handles point to point turtlebot motion, avoiding obstacles.
    It is the sole node that should publish to cmd_vel
    """

    def __init__(self):
        rospy.init_node("turtlebot_navigator", anonymous=True)
        self.mode = Mode.IDLE

        # current state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # goal state
        self.x_g = None
        self.y_g = None
        self.theta_g = None

        self.th_init = 0.0

        # map parameters
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0
        self.map_origin = [0, 0]
        self.map_probs = []
        self.occupancy = None
        self.occupancy_updated = False

        # plan parameters
        self.plan_resolution = 0.05 #0.1 #CHANGED RESOLUTION
        self.plan_horizon = 6 #15 #CHANGED plan size

        # time when we started following the plan
        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = 0
        self.plan_start = [0.0, 0.0]

        # Robot limits
        self.v_max = 0.2  # maximum velocity
        self.om_max = 0.4  # maximum angular velocity

        self.v_des = 0.12  # desired cruising velocity
        self.theta_start_thresh = 0.05  # threshold in theta to start moving forward when path-following
        self.start_pos_thresh = (
            0.2  # threshold to be far enough into the plan to recompute it
        )

        # threshold at which navigator switches from trajectory to pose control
        self.near_thresh = 0.2 #0.2
        self.at_thresh = 0.05 #0.1
        self.at_thresh_theta = 2*np.pi #0.1
        self.align_at_thresh_theta = np.pi/16 #0.1

        # trajectory smoothing
        self.spline_alpha = 0.01 #original 0.15
        self.traj_dt = 0.1

        # trajectory tracking controller parameters
        self.kpx = 0.5
        self.kpy = 0.5
        self.kdx = 1.5
        self.kdy = 1.5

        #HERE distance for choosing robo pose next to object
        self.distance_stop_sign = .5
        self.distance_fire_hydrant = .5
        self.distance_person = .5
        self.distance_cow = .5

        self.confidence_stop_sign = .9
        self.confidence_fire_hydrant = .9
        self.confidence_person = .9
        self.confidence_cow = .9

        # heading controller parameters
        self.kp_th = 2.0

        self.traj_controller = TrajectoryTracker(
            self.kpx, self.kpy, self.kdx, self.kdy, self.v_max, self.om_max
        )
        self.pose_controller = PoseController(
            1, 1, 1, self.v_max, self.om_max #changed from passing PoseController 0,0,0 for gains
        )
        self.heading_controller = HeadingController(self.kp_th, self.om_max)

        self.nav_planned_path_pub = rospy.Publisher(
            "/planned_path", Path, queue_size=10
        )
        self.nav_smoothed_path_pub = rospy.Publisher(
            "/cmd_smoothed_path", Path, queue_size=10
        )
        self.nav_smoothed_path_rej_pub = rospy.Publisher(
            "/cmd_smoothed_path_rejected", Path, queue_size=10
        )
        self.nav_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        ## MY PUBS for marker
        self.nav_marker_pub = rospy.Publisher('marker_topic', Marker, queue_size=10)

        self.trans_listener = tf.TransformListener()

        self.cfg_srv = Server(NavigatorConfig, self.dyn_cfg_callback)

        #save the locations of objects when they are found, and hardcode waypoints to drive to in the exploration phase
        self.object_locations = {
            "waypoint1":Pose2D(3.349561400377962, 2.8039031395460303, 3.134128448458997),
            "waypoint2":Pose2D(2.4108785357169795, 2.7958572574313045, -1.5758450241372495),
            "waypoint3":Pose2D(0.42593046167582843, 2.534268677132602, 3.0380163235476796),
            #"waypoint3":Pose2D(0.4913060136380367, 2.64188515596929, -2.4610471132775373),
            #"waypoint3":Pose2D(0.35094396586434196, 2.4560829863909333, -2.8887995713389585),
            #"waypoint4":Pose2D(0.6670895495163305, 1.6721748736921203, -0.024545853193942816),
            "waypoint4":Pose2D(0.2810136284600561, 1.6902888085199776, -0.0543469585481558),
            "waypoint5":Pose2D(2.340990731473792, 1.0121038599364363, -3.1302328189560864),
            "waypoint6":Pose2D(0.3674399494391384, 0.37599779711013137, -2.8082224223552004),
            "waypoint7":Pose2D(2.9731959141224227, 0.2658792083280731, 1.5775341253555768),
            "waypoint8":Pose2D(2.1605970857792336, 1.8262811878290115, 2.2486272576930553),
            "home":Pose2D(), "stop_sign":Pose2D(), "fire_hydrant":Pose2D(), "person":Pose2D(), "cow":Pose2D()}
        #"waypoint1":Pose2D(1,2,3),
        self.home_set = False

        rospy.Subscriber("/map_dilated", OccupancyGrid, self.map_callback) #THERE changed from map to map_dilated
        rospy.Subscriber("/map_metadata", MapMetaData, self.map_md_callback)
        rospy.Subscriber("/cmd_nav", Pose2D, self.cmd_nav_callback)
        rospy.Subscriber("/cmd_goal", String, self.cmd_goal_callback) # rostopic pub cmd_goal std_msgs/String person

        # HERE in order to save location when object is identified, need to know when object is identified
        rospy.Subscriber("/detector/stop_sign", DetectedObject, self.detector_stop_sign_callback)
        rospy.Subscriber("/detector/fire_hydrant", DetectedObject, self.detector_fire_hydrant_callback)
        rospy.Subscriber("/detector/person", DetectedObject, self.detector_person_callback)
        rospy.Subscriber("/detector/cow", DetectedObject, self.detector_cow_callback)

        #HERE in order to publish location need to use navigator.py as a node
        self.stop_sign_pub = rospy.Publisher("/stop_sign", Pose2D, queue_size=10)
        self.fire_hydrant_pub = rospy.Publisher("/fire_hydrant", Pose2D, queue_size=10)
        self.person_pub = rospy.Publisher("/person", Pose2D, queue_size=10)
        self.cow_pub = rospy.Publisher("/cow", Pose2D, queue_size=10)

        #RECOVER subscribe to scan topic to obtain Lidar information
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)

        #RECOVER category level variable to store the lidar information
        self.lidar_angle_dist_min = None
        #recovery angle value
        self.angle_recover = None

        print("finished init")

    def dyn_cfg_callback(self, config, level):
        rospy.loginfo(
            "Reconfigure Request: k1:{k1}, k2:{k2}, k3:{k3}".format(**config)
        )
        self.pose_controller.k1 = config["k1"]
        self.pose_controller.k2 = config["k2"]
        self.pose_controller.k3 = config["k3"]
        
        # ADDED Dynamically configurable params for detection params
        self.distance_stop_sign = config["distance_stop_sign"]
        self.distance_fire_hydrant = config["distance_fire_hydrant"]
        self.distance_person = config["distance_person"]
        self.distance_cow = config["distance_cow"]

        self.confidence_stop_sign = config["confidence_stop_sign"]
        self.confidence_fire_hydrant = config["confidence_fire_hydrant"]
        self.confidence_person = config["confidence_person"]
        self.confidence_cow = config["confidence_cow"]
        
        return config

    def detector_to_world_location(self, msg):
        location = Pose2D()
        location.theta = self.theta + (wrapToPi(msg.thetaleft) + wrapToPi(msg.thetaright))/2
        location.x = self.x + msg.distance * np.cos(location.theta)
        location.y = self.y + msg.distance * np.sin(location.theta)
        return location

    #HERE when object is detected, publish the position of the robot
    def detector_stop_sign_callback(self, msg):
        if msg.confidence > 0.80:
            if msg.distance < self.distance_stop_sign:
                # if abs(wrapToPi(msg.thetaleft)) < np.pi/6:
                #     if abs(wrapToPi(msg.thetaright)) < np.pi/6:
                if self.mode == Mode.IDLE: 
                    #marker
                    marker_location = self.detector_to_world_location(msg)
                    self.detector_marker(marker_location,1,'red')
                    #robo position
                    self.distance_stop_sign = msg.distance
                    location = Pose2D()
                    location.x = self.x
                    location.y = self.y
                    location.theta = self.theta
                    self.stop_sign_pub.publish(location)
                    #print("location", location)
                    self.object_locations["stop_sign"] = location

    def detector_fire_hydrant_callback(self, msg):
        if msg.confidence > 0.5: #0.98
            if msg.distance < self.distance_fire_hydrant:
                if self.mode == Mode.IDLE:  
                    #marker
                    marker_location = self.detector_to_world_location(msg)
                    self.detector_marker(marker_location,2)
                    #robo position
                    self.distance_fire_hydrant = msg.distance
                    location = Pose2D()
                    location.x = self.x
                    location.y = self.y
                    location.theta = self.theta
                    self.fire_hydrant_pub.publish(location) 
                    self.object_locations["fire_hydrant"] = location

    def detector_cardboard_box_callback(self, msg):
        if msg.confidence > 0.98:
            if msg.distance < self.distance_cardboard_box:
                if self.mode == Mode.IDLE:  
                    #marker
                    marker_location = self.detector_to_world_location(msg)
                    self.detector_marker(marker_location,3)
                    #robo position
                    self.distance_cardboard_box = msg.distance
                    location = Pose2D()
                    location.x = self.x
                    location.y = self.y
                    location.theta = self.theta
                    self.cardboard_box_pub.publish(location) 
                    self.object_locations["cardboard_box"] = location
    
    def detector_person_callback(self, msg):
        if msg.confidence > 0.9:
            if msg.distance < self.distance_person:
                if self.mode == Mode.IDLE:  
                    #marker
                    marker_location = self.detector_to_world_location(msg)
                    self.detector_marker(marker_location,4,'gb')
                    #robo position
                    self.distance_person = msg.distance
                    location = Pose2D()
                    location.x = self.x
                    location.y = self.y
                    location.theta = self.theta
                    self.person_pub.publish(location) 
                    self.object_locations["person"] = location

    def detector_cow_callback(self, msg):
        if msg.confidence > 0.5:
            if msg.distance < self.distance_cow:
                if self.mode == Mode.IDLE:
                    #robo position
                    self.distance_cow = msg.distance
                    location = Pose2D()
                    location.x = self.x
                    location.y = self.y
                    location.theta = self.theta
                    self.cow_pub.publish(location) 
                    self.object_locations["cow"] = location
                    #marker
                    marker_location = self.detector_to_world_location(msg)
                    self.detector_marker(marker_location,5,'purple')

    #RECOVER determine the minimum distance and its angle and send to recover if needed
    def scan_callback(self, msg):
        if self.mode == Mode.TRACK or self.mode == Mode.PARK:
            if msg.range_min > 0.2:
                return
            else:
                #the increment for this LaserScan is by ONE DEGREE
                lidar_range = np.array(msg.ranges)
                lidar_range[20:-20] += 100
                #print("limited view array", lidar_range)

                #find the minimum value within our looking range
                index = np.argmin(lidar_range)    
                #print("index of smallest range", index, "minimum distance", np.min(lidar_range))
                
                if np.min(lidar_range) < 0.2:
                    angle_start = msg.angle_min
                    angle_increment = msg.angle_increment
                    self.lidar_angle_dist_min = angle_start + index * angle_increment 
                    self.angle_recover = self.lidar_angle_dist_min + np.pi + self.theta
                    #print("calcualted angle",  self.lidar_angle_dist_min)   
                    self.switch_mode(Mode.RECOVER)

    def cmd_goal_callback(self, data):
        # print(data, self.object_locations[data])
        print(self.object_locations[data.data])
        self.cmd_nav_callback(self.object_locations[data.data])

    def cmd_nav_callback(self, data):
        """
        loads in goal if different from current goal, and replans
        """
        if (
            data.x != self.x_g
            or data.y != self.y_g
            or data.theta != self.theta_g
        ):
            self.x_g = data.x
            self.y_g = data.y
            self.theta_g = data.theta
            self.replan()

            
            #MY CHANGES for marker
            self.detector_marker(data, 0, 'green')
            ## END MY CHANGES

    def detector_marker(self, data, id, color = 'blue'):
        #MY CHANGES for marker
        marker = Marker()

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time()

        # IMPORTANT: If you're creating multiple markers, 
        #            each need to have a separate marker ID.
        marker.id = id

        marker.type = 2 # sphere

        marker.pose.position.x = data.x
        marker.pose.position.y = data.y
        marker.pose.position.z = .1

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = data.theta
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        marker.color.a = 1.0 # Don't forget to set the alpha!
        if color == 'red':
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        elif color == 'green':
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        elif color == 'blue':
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
        elif color == 'bg':
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
        elif color == 'purple':
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        else:
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
    
        
        self.nav_marker_pub.publish(marker)
        ## END MY CHANGES

    def map_md_callback(self, msg):
        """
        receives maps meta data and stores it
        """
        self.map_width = msg.width
        self.map_height = msg.height
        self.map_resolution = msg.resolution
        self.map_origin = (msg.origin.position.x, msg.origin.position.y)

    def map_callback(self, msg):
        """
        receives new map info and updates the map
        """
        #print("entered map_callback")
        self.map_probs = msg.data
        # if we've received the map metadata and have a way to update it:
        if (
            self.map_width > 0
            and self.map_height > 0
            and len(self.map_probs) > 0
        ):
            self.occupancy = StochOccupancyGrid2D(
                self.map_resolution,
                self.map_width,
                self.map_height,
                self.map_origin[0],
                self.map_origin[1],
                3, #changed from 1
                self.map_probs,
            )
           # print("successful occupancy set")
            if self.x_g is not None:
                # if we have a goal to plan to, replan
                rospy.loginfo("replanning because of new map")
                self.replan()  # new map, need to replan

    def shutdown_callback(self):
        """
        publishes zero velocities upon rospy shutdown
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.nav_vel_pub.publish(cmd_vel)

    def near_goal(self):
        """
        returns whether the robot is close enough in position to the goal to
        start using the pose controller
        """
        print("x, y", self.x, self.y)
        print("x_g, y_g", self.x_g, self.y_g)
        return (
            linalg.norm(np.array([self.x - self.x_g, self.y - self.y_g]))
            < self.near_thresh
        )

    def at_goal(self):
        """
        returns whether the robot has reached the goal position with enough
        accuracy to return to idle state
        """
        return (
            linalg.norm(np.array([self.x - self.x_g, self.y - self.y_g]))
            < self.at_thresh
            and abs(wrapToPi(self.theta - self.theta_g)) < self.at_thresh_theta
        )

    def aligned(self):
        """
        returns whether robot is aligned with starting direction of path
        (enough to switch to tracking controller)
        """
        return (
            abs(wrapToPi(self.theta - self.th_init)) < self.theta_start_thresh
        )

    #new definition for success in alignment for parking
    def park_aligned(self):
        """
        returns whether robot is aligned with goal theta
        (enough to pause)
        """
        return (
            abs(wrapToPi(self.theta - self.theta_g)) < self.align_at_thresh_theta #CHANGED from self.at_thresh_theta 
        )

    #RECOVER new definition for success in alignment for recovery
    def recover(self):
        """
        returns whether robot is aligned with goal theta
        (enough to pause)
        """
        return (
            abs(wrapToPi(self.theta - self.angle_recover)) < self.align_at_thresh_theta #CHANGED from self.at_thresh_theta 
        )

    def close_to_plan_start(self):
        return (
            abs(self.x - self.plan_start[0]) < self.start_pos_thresh
            and abs(self.y - self.plan_start[1]) < self.start_pos_thresh
        )

    def snap_to_grid(self, x):
        return (
            self.plan_resolution * round(x[0] / self.plan_resolution),
            self.plan_resolution * round(x[1] / self.plan_resolution),
        )

    def switch_mode(self, new_mode):
        rospy.loginfo("Switching from %s -> %s", self.mode, new_mode)
        self.mode = new_mode

    def publish_planned_path(self, path, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = "map"
        for state in path:
            pose_st = PoseStamped()
            pose_st.pose.position.x = state[0]
            pose_st.pose.position.y = state[1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = "map"
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_smoothed_path(self, traj, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = "map"
        for i in range(traj.shape[0]):
            pose_st = PoseStamped()
            pose_st.pose.position.x = traj[i, 0]
            pose_st.pose.position.y = traj[i, 1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = "map"
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_control(self):
        """
        Runs appropriate controller depending on the mode. Assumes all controllers
        are all properly set up / with the correct goals loaded
        """
        t = self.get_current_plan_time()

        if self.mode == Mode.PARK:
            V, om = self.pose_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        elif self.mode == Mode.TRACK:
            V, om = self.traj_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        elif self.mode == Mode.ALIGN:
            V, om = self.heading_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        elif self.mode == Mode.PARK_ALIGN:
            self.heading_controller.load_goal(self.theta_g)
            V, om = self.heading_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        elif self.mode == Mode.RECOVER:
            self.heading_controller.load_goal(self.angle_recover)
            _, om = self.heading_controller.compute_control(
                self.x, self.y, self.theta, t
            )
            V = 0.0
        else:
            V = 0.0
            om = 0.0

        cmd_vel = Twist()
        cmd_vel.linear.x = V
        cmd_vel.angular.z = om
        self.nav_vel_pub.publish(cmd_vel)

    def get_current_plan_time(self):
        t = (rospy.get_rostime() - self.current_plan_start_time).to_sec()
        return max(0.0, t)  # clip negative time to 0

    def replan(self):
        """
        loads goal into pose controller
        runs planner based on current pose
        if plan long enough to track:
            smooths resulting traj, loads it into traj_controller
            sets self.current_plan_start_time
            sets mode to ALIGN
        else:
            sets mode to PARK
        """
        if self.mode == Mode.PAUSE or self.mode == Mode.RECOVER: #add not in pause condition otherwise exits pause
            print("####Mode is Pause or Recover, returning from replan")
            return

        # Make sure we have a map
        if not self.occupancy:
            rospy.loginfo(
                "Navigator: replanning canceled, waiting for occupancy map."
            )
            self.switch_mode(Mode.IDLE)
            return

        # Attempt to plan a path
        state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
        state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
        x_init = self.snap_to_grid((self.x, self.y))
        self.plan_start = x_init
        x_goal = self.snap_to_grid((self.x_g, self.y_g))
        problem = AStar(
            state_min,
            state_max,
            x_init,
            x_goal,
            self.occupancy,
            self.plan_resolution,
        )

        rospy.loginfo("Navigator: computing navigation plan")
        success = problem.solve()
        if not success:
            rospy.loginfo("Planning failed")
            return
        else:
            rospy.loginfo("Planning Succeeded")
            self.x_g = problem.x_goal[0]
            self.y_g = problem.x_goal[1]            

        planned_path = problem.path

        # Check whether path is too short
        if len(planned_path) < 4:
            rospy.loginfo("Path too short to track")
            self.switch_mode(Mode.PARK)
            return

        # Smooth and generate a trajectory
        traj_new, t_new = compute_smoothed_traj(
            planned_path, self.v_des, self.spline_alpha, self.traj_dt
        )

        # If currently tracking a trajectory, check whether new trajectory will take more time to follow
        if self.mode == Mode.TRACK:
            t_remaining_curr = (
                self.current_plan_duration - self.get_current_plan_time()
            )

            # Estimate duration of new trajectory
            th_init_new = traj_new[0, 2]
            th_err = wrapToPi(th_init_new - self.theta)
            t_init_align = abs(th_err / self.om_max)
            t_remaining_new = t_init_align + t_new[-1]

            if t_remaining_new > t_remaining_curr:
                rospy.loginfo(
                    "New plan rejected (longer duration than current plan)"
                )
                self.publish_smoothed_path(
                    traj_new, self.nav_smoothed_path_rej_pub
                )
                return

        # Otherwise follow the new plan
        self.publish_planned_path(planned_path, self.nav_planned_path_pub)
        self.publish_smoothed_path(traj_new, self.nav_smoothed_path_pub)

        self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
        self.traj_controller.load_traj(t_new, traj_new)

        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = t_new[-1]

        self.th_init = traj_new[0, 2]
        self.heading_controller.load_goal(self.th_init)

        if not self.aligned():
            rospy.loginfo("Not aligned with start direction")
            self.switch_mode(Mode.ALIGN)
            return

        rospy.loginfo("Ready to track")
        self.switch_mode(Mode.TRACK)

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # try to get state information to update self.x, self.y, self.theta
            try:
                (translation, rotation) = self.trans_listener.lookupTransform(
                    "/map", "/base_footprint", rospy.Time(0)
                )
                self.x = translation[0]
                self.y = translation[1]
                euler = tf.transformations.euler_from_quaternion(rotation)
                self.theta = euler[2]
                if not self.home_set:
                    self.object_locations["home"] = Pose2D(self.x, self.y, self.theta)
                    self.home_set = True
            except (
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
            ) as e:
                self.current_plan = []
                rospy.loginfo("Navigator: waiting for state info")
                self.switch_mode(Mode.IDLE)
                print(e)
                pass

            # STATE MACHINE LOGIC
            # some transitions handled by callbacks
            if self.mode == Mode.IDLE:
                pass
            elif self.mode == Mode.ALIGN:
                if self.aligned():
                    self.current_plan_start_time = rospy.get_rostime()
                    self.switch_mode(Mode.TRACK)            
            elif self.mode == Mode.TRACK:
                if self.near_goal():
                    self.switch_mode(Mode.PARK)
                elif not self.close_to_plan_start():
                    rospy.loginfo("replanning because far from start")
                    self.replan()
                elif (
                    rospy.get_rostime() - self.current_plan_start_time
                ).to_sec() > self.current_plan_duration:
                    rospy.loginfo("replanning because out of time")
                    self.replan()  # we aren't near the goal but we thought we should have been, so replan
            elif self.mode == Mode.PARK:
                if self.at_goal():
                    self.switch_mode(Mode.PARK_ALIGN)
            elif self.mode == Mode.PARK_ALIGN:
                if self.park_aligned():
                    self.start_time = rospy.get_rostime()
                    self.switch_mode(Mode.PAUSE)
            elif self.mode == Mode.PAUSE:
                self.x_g = None
                self.y_g = None
                self.theta_g = None
                if (rospy.get_rostime() - self.start_time) > rospy.Duration.from_sec(3):
                    rospy.loginfo("######### DONE #########")
                    #forget about goal:
                    #self.x_g = None
                    #self.y_g = None
                    #self.theta_g = None
                    self.switch_mode(Mode.IDLE)
            elif self.mode == Mode.RECOVER:
                if self.recover():
                    self.current_plan_start_time = rospy.get_rostime()
                    self.switch_mode(Mode.IDLE) #when replan is called, if goal exists, replan will send to ALIGN
                    self.replan()  # we turned around now find a new path
                    
                else:
                    #rospy.loginfo("!######## PAUSED ########!")
                    pass

            self.publish_control()
            rate.sleep()


if __name__ == "__main__":
    nav = Navigator()
    rospy.on_shutdown(nav.shutdown_callback)
    nav.run()
