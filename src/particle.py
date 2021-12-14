import random
from math import atan2
from typing import List, Tuple

import numpy as np
from numpy import cos, sin, pi


class LandmarkEKF(object):
    """EKF representing a landmark"""

    def __init__(self):
        self.observed: bool = False
        self.mu: np.array = np.vstack([0, 0])  # landmark position as vector of length 2
        self.sigma: np.array = np.zeros((2, 2))  # covariance as 2x2 matrix

    def __str__(self):
        return "LandmarkEKF(observed  = {0}, mu = {1}, sigma = {2})".format(
            self.observed, self.mu, self.sigma
        )


class Particle(object):
    """Particle for tracking a robot with a particle filter.

    The particle consists of:
    - a robot pose
    - a weight
    - a map consisting of landmarks
    """

    def __init__(self, num_particles, num_landmarks, noise):
        """Creates the particle and initializes location/orientation"""
        self.noise = noise

        # initialize robot pose at origin
        self.pose = np.vstack([0.0, 0.0, 0.0])

        # initialize weights uniformly
        self.weight = 1.0 / float(num_particles)

        # Trajectory of the particle
        self.trajectory = []

        # initialize the landmarks aka the map
        self.landmarks = [LandmarkEKF() for _ in range(num_landmarks)]

    def prediction_step(self, odom):
        """Predict the new pose of the robot"""

        # append the old position
        self.trajectory.append(self.pose)

        # noise sigma for delta_rot1
        delta_rot1_noisy = random.gauss(odom.r1, self.noise[0])

        # noise sigma for translation
        translation_noisy = random.gauss(odom.t, self.noise[1])

        # noise sigma for delta_rot2
        delta_rot2_noisy = random.gauss(odom.r2, self.noise[2])

        # Estimate of the new position of the Particle
        x_new = self.pose[0] + translation_noisy * cos(self.pose[2] + delta_rot1_noisy)
        y_new = self.pose[1] + translation_noisy * sin(self.pose[2] + delta_rot1_noisy)
        theta_new = normalize_angle(self.pose[2] + delta_rot1_noisy + delta_rot2_noisy)

        self.pose = np.vstack([x_new, y_new, theta_new])

    def correction_step(self, sensor_measurements):
        """Weight the particles according to the current map of the particle and the landmark observations z.

        - sensor_measurements                : list of sensor measurements for the current timestep
        - sensor_measurements[i].landmark_id : observed landmark ID
        - sensor_measurements[i].z_range     : measured distance to the landmark in meters
        - sensor_measurements[i].z_bearing   : measured angular direction of the landmark in radians
        """

        # Construct the sensor noise matrix Q (2 x 2)
        Q_t = 0.1 * np.identity(2)

        robot_pose = self.pose

        # process each sensor measurement
        for measurement in sensor_measurements:
            # Get the EKF representing the landmark of the current observation
            landmark = self.landmarks[measurement.landmark_id]

            # The (2x2) EKF of the landmark is given by
            # its mean landmarks[landmark_id].mu
            # and by its covariance landmarks[landmark_id].sigma
            
            # If the landmark is observed for the first time:
            if not landmark.observed:
                # TODO: Initialize its position based on the measurement and the current Particle pose:
                # the absolute bearing in radians
                absolute_bearing = robot_pose[2] + measurement.z_bearing
                landmark_x = self.pose[0] + np.sin(absolute_bearing) * measurement.z_range
                landmark_y = self.pose[1] + np.cos(absolute_bearing) * measurement.z_range

                # get the Jacobian
                [h, H] = self.measurement_model(landmark)

                # TODO: initialize the EKF for this landmark
                landmark.mu = np.vstack([landmark_x, landmark_y])
                landmark.sigma = np.matmul(
                    np.matmul(np.linalg.inv(H), Q_t), np.transpose(np.linalg.inv(H))
                )  # see exercise description

                # Indicate that this landmark has been observed
                landmark.observed = True

            else:
                # get the expected measurement and the Jacobian
                expected_z, H = self.measurement_model(landmark)

                # TODO: compute the measurement covariance
                s_t: np.array = (
                    np.matmul(np.matmul(H, np.conjugate(landmark.sigma)), np.transpose(H))
                    + Q_t
                )

                # TODO: calculate the Kalman gain
                K_t = np.matmul(
                    np.matmul(np.conj(landmark.sigma), np.transpose(H)),
                    np.linalg.inv(s_t),
                )

                # compute the error between the z and expected_z (remember to normalize the angle)
                error = [measurement.z_range - expected_z[0], normalize_angle(measurement.z_bearing) - expected_z[1]]

                # update the mean and covariance of the EKF for this landmark
                landmark.mu = np.conj(landmark.mu) + np.matmul(K_t, error)
                landmark.sigma = np.matmul(
                    (np.identity(2) - np.matmul(K_t, H)), np.conjugate(landmark.sigma)
                )

                # compute the likelihood of this observation, multiply with the former weight
                # to account for observing several features in one time step
                self.weight = self.get_probability(error) * self.weight

    def get_probability(self, error) -> float:
        distance_prob = np.exp((-error[0]**2)/2) / np.sqrt(2*np.pi)
        angle_prob = np.exp((-error[1]**2)/2) / np.sqrt(2*np.pi)
        return distance_prob * angle_prob

    def measurement_model(self, landmark_ekf) -> Tuple[List[float], np.array]:
        """Compute the expected measurement for a landmark and the Jacobian

        - landmark_ekf: EKF representing the landmark

        Returns a tuple (h, H) where
        - h = [expected_range, expected_range] is the expected measurement
        - H is the Jacobian.
        """
        # position (x, y) of the observed landmark_ekf
        landmark_x = landmark_ekf.mu[0]
        landmark_y = landmark_ekf.mu[1]

        # use the current state of the particle to predict the measurement
        expected_range = np.sqrt(
            (landmark_x - self.pose[0]) ** 2 + (landmark_y - self.pose[1]) ** 2
        )
        angle = (
            atan2(landmark_y - self.pose[1], landmark_x - self.pose[0]) - self.pose[2]
        )
        expected_bearing = normalize_angle(angle)

        h = [expected_range, expected_bearing]

        # Compute the Jacobian H of the measurement z with respect to the landmark position
        H = np.zeros((2, 2))
        H[0, 0] = (landmark_x - self.pose[0]) / expected_range  # d_range / d_lx
        H[0, 1] = (landmark_y - self.pose[1]) / expected_range  # d_range / d_ly

        H[1, 0] = (self.pose[1] - landmark_y) / (
            expected_range ** 2
        )  # d_bearing / d_lx
        H[1, 1] = (landmark_x - self.pose[0]) / (
            expected_range ** 2
        )  # d_bearing / d_ly

        return h, H


def normalize_angle(angle):
    """Normalize the angle between -pi and pi"""

    while angle > pi:
        angle = angle - 2.0 * pi

    while angle < -pi:
        angle = angle + 2.0 * pi

    return angle
