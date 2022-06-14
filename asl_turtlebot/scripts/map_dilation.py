#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
import numpy as np
import scipy.signal

class Dilation:
    """This node handles map dilation"""

    def __init__(self):
        # Initialize ROS node
        rospy.init_node("map_dilation", anonymous=True)
        #create map_dilated the same size as map
        self.map_dilated = OccupancyGrid()

        #subscribe to map
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)

        #publish to map_dilated topic
        self.map_dilation_pub = rospy.Publisher("/map_dilated", OccupancyGrid, queue_size=10)

    def map_callback(self, msg):

        #set the header and metadata for map_dilated
        self.map_dilated.header = msg.header
        self.map_dilated.info = msg.info
        #print("map_dilated.header", self.map_dilated.header)
        #print("map_dilated.info", self.map_dilated.info)

        #unpack msg data to make it easier to manipulate
        I = msg.data #I stands for "image"
        #print("Length map.data", len(I))

        #change I from tuple to array for easier manipulation
        I = np.asarray(I)

        #reshape I from a single vector (row major) to row and column
        I = I.reshape(self.map_dilated.info.width,self.map_dilated.info.height)
        #print("Shape reshaped I array", I.shape, I.size)

        #create the dilation filter
        F = np.ones((5,5))
        F = F #/ 25 #map is made of Int8, so values above 257 would wrap around

        F[2][2] = 0

        # print("F", F)

        #convolute the arrays (given a symmetric array across the horizontal and verticla, this is same as correlateion)
        dilated_array = (scipy.signal.fftconvolve(I, F, 'same'))
        #print("Shape dilated_array", dilated_array.shape, dilated_array.size)
        #print("array type", type(dilated_array[0][0]))

        #reshape dilated_array from row and column to a single vector (row major)
        dilated_array = dilated_array.reshape(dilated_array.size)
        #dilated_array = (dilated_array * 100 / np.max(dilated_array)).astype(np.int8)
        # dilated_array = np.where(dilated_array < 0, -2, dilated_array)
        dilated_array = np.where(dilated_array > 1, 100, dilated_array).astype(np.int8)
        #print(np.min(dilated_array), np.max(dilated_array))
        #print("Shape resized dilated_array", dilated_array.shape, dilated_array.size)

        #turn the array back into a tuple
        dilated_tuple = tuple(dilated_array)
        #print("Length dilated_tuple", len(dilated_tuple))

        #set the data property of the dilated map to the tuple containing the convoluted map
        self.map_dilated.data = dilated_tuple
        #print("msg data type", type(msg.data[0]))
        #print("dilated data type", type(self.map_dilated.data[0]))
        #print("tuple type", type(dilated_tuple[0]))

        #publish to map_dilated topic
        self.map_dilation_pub.publish(self.map_dilated)
        # print("new published")

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    dilation = Dilation()
    dilation.run()