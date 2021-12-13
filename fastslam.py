#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FastSLAM algorithm for range-bearing landmark observations.
"""

from __future__ import print_function, division

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

from src.measurement import SensorMeasurement
from src.odometry import Odometry
from src.particle import Particle
from src.plotter import Plotter
from src.timestep import TimeStep


def main():
    """Main function of the program.

    This script calls all the required functions in the correct order.
    You can change the number of steps the filter runs for to ease the
    debugging. You should however not change the order or calls of any
    of the other lines, as it might break the framework.

    If you are unsure about the input and return values of functions you
    should read their documentation which tells you the expected dimensions.
    """

    # Read world data, i.e. landmarks. The true landmark positions are not given to the Particle
    landmarks_truth = read_world("data/world.dat")

    # Read sensor readings, i.e. odometry and range-bearing sensor
    data = read_sensor_data("data/sensor_data.dat")

    # how many particles
    num_particles = 100

    # Get the number of landmarks in the map
    num_landmarks = len(landmarks_truth)

    noise = [0.005, 0.01, 0.005]

    # initialize the particles array
    particles = [
        Particle(num_particles, num_landmarks, noise) for _ in range(num_particles)
    ]

    # set the axis dimensions
    plotter = Plotter(data, particles, landmarks_truth)

    anim = animation.FuncAnimation(
        plt.gcf(),
        plotter.fast_slam,
        frames=np.arange(0, len(data)),
        init_func=plotter.plot_state_init,
        interval=20,
        blit=True,
        repeat=False,
    )
    if anim:
        plt.show()


def read_world(filename):
    """Reads the world definition and returns a structure of landmarks.

    filename: path of the file to load
    landmarks: structure containing the parsed information

    Each landmark contains the following information:
    - id : id of the landmark
    - x  : x-coordinate
    - y  : y-coordinate

    Examples:
    - Obtain x-coordinate of the 5-th landmark
      landmarks(5).x
    """
    landmarks = {}
    with open(filename) as world_file:
        for line in world_file:
            line_s = line.rstrip()  # remove newline character

            line_spl = line_s.split(" ")

            landmarks[int(line_spl[0])] = [float(line_spl[1]), float(line_spl[2])]

    return landmarks


def read_sensor_data(filename):
    """Reads the odometry and sensor readings from a file.

    filename: path to the file to parse

    The data is returned as an array of time steps, each time step containing
    odometry data and sensor measurements.

    Usage:
    - access the readings for timestep t:
      data[t]
      this returns a TimeStep object containing the odometry reading and all
      landmark observations, which can be accessed as follows
    - odometry reading at timestep t:
      data[i].odometry
    - sensor reading at timestep i:
      data[i].sensor

    Odometry readings have the following fields:
    - r1 : initial rotation in radians counterclockwise
    - t  : translation in meters
    - r2 : final rotation in radians counterclockwise
    which correspond to the identically labeled variables in the motion
    mode.

    Sensor readings can again be indexed and each of the entries has the
    following fields:
    - landmark_id : id of the observed landmark
    - z_range     : measured range to the landmark in meters
    - z_bearing   : measured angle to the landmark in radians

    Examples:
    - Translational component of the odometry reading at timestep 10
      data[10].odometry.t
    - Measured range to the second landmark observed at timestep 4
      data[4].sensor[1].z_range
    """
    data = []
    sensor_measurements = []
    first_time = True
    odom = None
    with open(filename) as data_file:
        for line in data_file:
            line_s = line.rstrip()  # remove the new line character
            line_spl = line_s.split(" ")  # split the line
            if line_spl[0] == "ODOMETRY":
                if not first_time:
                    data.append(TimeStep(odom=odom, sensor=sensor_measurements))
                    sensor_measurements = []
                first_time = False
                odom = Odometry(
                    r1=float(line_spl[1]), t=float(line_spl[2]), r2=float(line_spl[3])
                )
            if line_spl[0] == "SENSOR":
                sensor_measurements.append(
                    SensorMeasurement(
                        landmark_id=int(line_spl[1]),
                        z_range=float(line_spl[2]),
                        z_bearing=float(line_spl[3]),
                    )
                )

    data.append(TimeStep(odom=odom, sensor=sensor_measurements))
    return data


if __name__ == "__main__":
    main()
