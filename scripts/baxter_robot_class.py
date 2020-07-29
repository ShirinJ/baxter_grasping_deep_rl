import roslib
import argparse
import math
import random
import rospy
import rospkg
import numpy
import sys
import time
import cv2
from cv_bridge import CvBridge
import message_filters

from sensor_msgs.msg import Image
from gazebo_msgs.srv import SetModelState

from std_msgs.msg import (
    UInt16,
    String,
    Int8
)

from geometry_msgs.msg import (
    Pose,
    Point,
    Quaternion,
)

from gazebo_msgs.msg import (
    ModelState,
    ModelStates,
    ContactState,
    ContactsState
)

import baxter_interface
from baxter_interface import CHECK_VERSION


class BaxterManipulator(object):

    def __init__(self):
        """
		BaxterManipulator control from torch - Initialisation
		"""
        # Publishers
        self._pub_rate = rospy.Publisher('robot/joint_state_publish_rate', UInt16, queue_size=10)
        self.image_pub = rospy.Publisher("baxter_view", Image, queue_size=4)
        self.dep_image_pub = rospy.Publisher("DepthMap", Image, queue_size=4)
        self.score_pub = rospy.Publisher("score", Int8, queue_size=10)

        self._obj_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        # Link with baxter interface
        self._left_arm = baxter_interface.limb.Limb("left")
        self._left_joint_names = self._left_arm.joint_names()
        self.grip_left = baxter_interface.Gripper('left', CHECK_VERSION)

        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

        # Control parameters
        self._rate = 500.0  # Hz
        self._pub_rate.publish(self._rate)
        self.bridge = CvBridge()
        self._left_arm.set_joint_position_speed(0.3)
        self._object_type = 0

    def position_gazebo_models(self):
        modelstate = ModelState()
        # return object to previous
        if self._object_type != 0:
            modelstate.model_name = "object" + str(self._object_type)
            modelstate.reference_frame = "world"
            modelstate.pose = Pose(position=Point(x=-3.0, y=0.0, z=0.0))
            req = self._obj_state(modelstate)

        # Randomise object type and orientation
        object_angle = 0.0  # random.uniform(0,math.pi)
        # orientation as quaternion
        object_q_z = math.sin(object_angle / 2)
        object_q_w = math.cos(object_angle / 2)
        # Position
        object_x = 0.625 + random.uniform(-0.1, 0.13)
        object_y = 0.7975 + random.uniform(-0.1, 0.13)
        # Type of object
        self._object_type = random.randint(1, 9)
        modelstate.model_name = "object" + str(self._object_type)

        # Place object for pick-up
        modelstate.reference_frame = "world"
        object_pose = Pose(
            position=Point(x=object_x, y=object_y, z=0.8),
            orientation=Quaternion(x=0.0, y=0.0, z=object_q_z, w=object_q_w)
        )

        modelstate.pose = object_pose
        req = self._obj_state(modelstate)

    def _reset(self):
        self.grip_left.open()

        rest_pos_left = [0.0, -0.55, 0.0, 0.75, 0.0, math.pi / 2.0 - 0.2, 0.0]
        # Set positions
        self._left_positions = dict(zip(self._left_arm.joint_names(), rest_pos_left))
        self._torques = dict(zip(self._left_arm.joint_names(), [0.0] * 7))

        self._left_arm.move_to_joint_positions(self._left_positions)
        # Reset models
        self.position_gazebo_models()

        # Reset task completion flag
        self._task_complete = 0
        # Set cmd timeout
        self._left_arm.set_command_timeout(0.5)

    def _reset_control_modes(self):
        rate = rospy.Rate(self._rate)
        for _ in xrange(100):
            if rospy.is_shutdown():
                return False
            self._left_arm.exit_control_mode()
            self._pub_rate.publish(100)  # 100Hz default joint state rate
            rate.sleep()
        return True

    def clean_shutdown(self):
        print("\nExiting example...")
        # return to normal
        self._left_arm.move_to_neutral()
        self._reset_control_modes()
        if not self._init_state:
            print("Disabling robot...")
            self._rs.disable()
        return True

    ''' Callback functions '''

    # Recieve action command from torch
    def action_callback(self, data):
        self.cmd = data.data
        self.action()

    # Recieve image from gazebo - resizes to 60 by 60
    def img_callback(self, img_data, rgb_data):
        self.cv_image = self.bridge.imgmsg_to_cv2(img_data, "rgba8")
        self.cv_image = self.cv_image[100:400, 300:600]
        self.cv_image = cv2.resize(self.cv_image, (60, 60))
        self.cv_image[:, :, 3] = 0;

        # Add motor angle information to alpha channel of image
        shoulder_angle = self._left_arm.joint_angle("left_s0")
        elbow_angle = self._left_arm.joint_angle("left_e1")

        # remove discontinuities by finding cosines and sines of angles
        self.cv_image[0, 0, 3] = int(255 * math.cos(elbow_angle) / (2.0 * math.pi))
        self.cv_image[0, 1, 3] = int(255 * math.sin(elbow_angle) / (2.0 * math.pi))
        self.cv_image[0, 2, 3] = int(255 * math.cos(shoulder_angle) / (2.0 * math.pi))
        self.cv_image[0, 3, 3] = int(255 * math.sin(shoulder_angle) / (2.0 * math.pi))

        self.cv_rgb_img = self.bridge.imgmsg_to_cv2(rgb_data, "rgba8")
        self.cv_rgb_img = self.cv_rgb_img[110:320, 55:265]  # crop away uninformative outer region of depth map
        self.cv_rgb_img = cv2.resize(self.cv_rgb_img, (60, 60))

    # Recieve object pose information - used to determine whether task has been completed
    def object_pose_callback(self, data):
        if self._object_type != 0:
            index = data.name.index('object' + str(self._object_type))
            self.object_position_x = data.pose[index].position.x
            self.object_position_y = data.pose[index].position.y
            self.object_position_z = data.pose[index].position.z

            # check for some velocity - if when picking up it hit the object - for an additional reward
            self.object_v = data.twist[index].angular.x  # just need any velocity - does not matter

    def listener(self):
        rospy.Subscriber('chatter', String, self.action_callback)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.object_pose_callback)
        self.img_sub = message_filters.Subscriber("/cameras/left_hand_camera/image", Image)
        self.dep_sub = message_filters.Subscriber("/DepCamera/image_raw", Image)
        self.timeSync = message_filters.ApproximateTimeSynchronizer([self.img_sub, self.dep_sub], 4, 0.01)
        self.timeSync.registerCallback(self.img_callback)

        rospy.spin()

    def move_vertical(self, direction):

        # Limb Lengths
        r1 = 0.37082
        r2 = 0.37442
        s1_offset = math.atan2(0.069, r1)  # offset due to structure of baxter arm at the e1 joint

        # Find x and y coordinates of w1 (wrist) joint from current s1 and e1 joint angles
        x = r1 * math.cos(self._left_positions["left_s1"] + s1_offset) \
            + r2 * math.cos(self._left_positions["left_s1"] + self._left_positions["left_e1"] + s1_offset)

        y = r1 * math.sin(self._left_positions["left_s1"] + s1_offset) \
            + r2 * math.sin(self._left_positions["left_s1"] + self._left_positions["left_e1"] + s1_offset)

        # Change y coordinate - move down to table or up to start position
        if direction == 'd':
            y = 0.2065
        elif direction == 'u':
            y = 0.0
        else:
            raise ValueError("Invalid movement")

        # Recalculate and set desired s1 and e1 angles
        cosine_theta2 = (math.pow(x, 2) + math.pow(y, 2) - math.pow(r1, 2) -
                         math.pow(r2, 2)) / (2 * r1 * r2)

        self._left_positions["left_e1"] = math.atan2(math.sqrt(abs(1 - math.pow(cosine_theta2, 2)) \
                                                               ), cosine_theta2)

        k1 = r1 + r2 * math.cos(self._left_positions["left_e1"])
        k2 = r2 * math.sin(self._left_positions["left_e1"])

        self._left_positions["left_s1"] = math.atan2(y, x) - math.atan2(k2, k1) - s1_offset

        # Keep w1 such that grip is vertical
        self._left_positions["left_w1"] = math.pi / 2.0 - self._left_positions["left_e1"] - \
                                          self._left_positions["left_s1"]

        self._pub_rate.publish(self._rate)
        self._left_arm.move_to_joint_positions(self._left_positions)

    def rotate_shoulder(self, direction):
        if direction == "right":
            self._left_positions["left_s0"] += 0.025
        elif direction == "left":
            self._left_positions["left_s0"] -= 0.025
        else:
            raise ValueError("Invalid movement")

        self._pub_rate.publish(self._rate)
        self._left_arm.set_joint_positions(self._left_positions)

    def adjust_reach(self, direction):
        if direction == "forward":
            self._left_positions["left_e1"] -= 0.05
            self._left_positions["left_s1"] += 0.05
        elif direction == "backwards":
            self._left_positions["left_e1"] += 0.05
            self._left_positions["left_s1"] -= 0.05
        else:
            raise ValueError("Invalid movement")

        # Adjustment to ensure gripper is verticle
        self._left_positions["left_w1"] = math.pi / 2.0 - self._left_positions["left_e1"] - self._left_positions[
            "left_s1"]
        self._pub_rate.publish(self._rate)
        self._left_arm.set_joint_positions(self._left_positions)

    def pick_up_object(self):
        obj_move = 0
        self.move_vertical("d")
        obj_contact_topic = "object_contact" + str(self._object_type)  # object_type needs to correspond with topic name in model urdf
        obj_contact = rospy.wait_for_message(obj_contact_topic, ContactsState)
        if self.object_v != 0:
            obj_move = 1
        time.sleep(0.2)
        self.grip_left.close()
        if self.object_v != 0:
            obj_move = 1
        time.sleep(1.0)
        self.move_vertical("u")
        if self.object_v != 0:
            obj_move = 1

        # Get effector endpoint pose - this is compared with the object pose
        # to determine if the task has been succesfully completed or not
        # z-axis of end affector adjusted by 1 due to difference in frame of reference
        self.end_x = self._left_arm.endpoint_pose()["position"].x
        self.end_y = self._left_arm.endpoint_pose()["position"].y
        self.end_z = 1.0 + self._left_arm.endpoint_pose()["position"].z

        # Comparison - threshold values are arbitrary, can be tweaked.
        if (abs(self.end_z - self.object_position_z) < 0.1):
            if (abs(self.end_y - self.object_position_y) < 0.055):
                if (abs(self.end_x - self.object_position_x) < 0.055):
                    self._task_complete = 10

        # Check for contact made with object for partial reward - for object pushed away
        elif obj_move != 0 or self.object_v != 0:
            self._task_complete = 1

        # Check for contact made with object for partial reward - for object not moving
        elif any((
                         contact.collision2_name == 'baxter::l_gripper_r_finger::l_gripper_r_finger_collision_l_gripper_r_finger_tip' or contact.collision2_name == 'baxter::l_gripper_l_finger::l_gripper_l_finger_collision_l_gripper_l_finger_tip')
                 for contact in obj_contact.states):
            self._task_complete = 1

        # Penalise unsuccessful attempts
        else:
            self._task_complete = -1

        self.score_pub.publish(self._task_complete)

    def action(self):
        if self.cmd == "1":
            self.rotate_shoulder("right")

        elif self.cmd == "2":
            self.rotate_shoulder("left")

        elif self.cmd == '3':
            self.adjust_reach("forward")

        elif self.cmd == '4':
            self.adjust_reach("backwards")

        elif self.cmd == '0':
            self.pick_up_object()

        elif self.cmd == 'reset':
            self._reset()

        # Publish image with terminal in header
        self.img_msg = self.bridge.cv2_to_imgmsg(self.cv_image, "rgba8")
        self.rgb_msg = self.bridge.cv2_to_imgmsg(self.cv_rgb_img, "rgba8")
        self.img_msg.header.frame_id = str(self._task_complete)
        self.image_pub.publish(self.img_msg)
        self.dep_image_pub.publish(self.rgb_msg)

