import copy
import random
from math import atan

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from numpy import pi, sqrt


# Chi square inverse cumulative distribution function for alpha = 0.95
# and 2 degrees of freedom (see lookup table in statistics textbook)
CHI_SQUARE_INV_95_2 = 5.99146


class Plotter(object):
    """Helper class for plotting the current state of the robot and map"""

    def __init__(self, data, particles, landmarks):
        self.data = data
        self.particles = particles
        self.landmarks = landmarks
        self.fig, self.ax = plt.subplots()
        plt.xlim(-2, 12)
        plt.ylim(-2, 12)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True)

        # plot the ground truth landmark positions as black crosses
        landmarks_x = [coordinates[0] for _, coordinates in self.landmarks.items()]
        landmarks_y = [coordinates[1] for _, coordinates in self.landmarks.items()]
        self.plot_landmarks = self.ax.plot(
            landmarks_x, landmarks_y, "k+", markersize=10, linewidth=5, animated=True
        )[0]

        # Plot the particles as green dots
        particles_x = [particle.pose[0] for particle in self.particles]
        particles_y = [particle.pose[1] for particle in self.particles]
        self.plot_particles = self.ax.plot(
            particles_x, particles_y, "g.", animated=True
        )[0]

        # draw the best particle as a red circle
        self.plot_best_particle = Ellipse(
            (0.0, 0.0), 0, 0, 0, color="r", fill=False, animated=True
        )
        self.ax.add_patch(self.plot_best_particle)

        # draw the trajectory as estimated by the currently best particle as a red line
        self.plot_trajectory = self.ax.plot(
            [0.0], [0.0], "r-", linewidth=3, animated=True
        )[0]

    def fast_slam(self, t):
        """Executes one iteration of the prediction-correction-resampling loop of FastSLAM.

        - t: Frame number of the current frame, starting with 0.

        Returns the plot objects to be drawn for the current frame.
        """
        # Perform filter update for each odometry-observation pair read from the data file.
        print("step {0}".format(t))

        # Perform the prediction step of the particle filter
        for particle in self.particles:
            particle.prediction_step(self.data[t].odom)

        # Perform the correction step of the particle filter
        for particle in self.particles:
            particle.correction_step(self.data[t].sensor)

        # Generate visualization plots of the current state of the filter
        r = self.plot_state(self.data[t].sensor)

        # Resample the particle set
        # Use the "number of effective particles" approach to resample only when
        # necessary. This approach reduces the risk of particle depletion.
        # For details, see Section IV.B of
        # http://www2.informatik.uni-freiburg.de/~burgard/postscripts/grisetti05icra.pdf
        s = sum([particle.weight for particle in self.particles])
        neff = 1.0 / sum([(particle.weight / s) ** 2 for particle in self.particles])
        if neff < len(self.particles) / 2.0:
            print("resample")
            self.particles = resample(self.particles)

        return r

    def plot_state_init(self):
        """Initializes the figure with the elements to be drawn"""
        return [self.plot_landmarks]

    def plot_state(self, sensor_measurements):
        """Visualizes the state of the FastSLAM algorithm.

        The resulting plot displays the following information:
        - map ground truth (black +'s)
        - currently best particle (red)
        - particle set in green
        - current landmark pose estimates (blue)
        - visualization of the observations made at this time step (line between Particle and landmark)
        """
        # update particle poses
        particles_x = [particle.pose[0] for particle in self.particles]
        particles_y = [particle.pose[1] for particle in self.particles]
        self.plot_particles.set_data(particles_x, particles_y)

        # determine the currently best particle
        weights = [particle.weight for particle in self.particles]
        index_of_best_particle = weights.index(max(weights))

        weight_sum = sum(weights)
        mean_pose = (
            sum(particle.weight * particle.pose for particle in self.particles)
            / weight_sum
        )
        cov_pose = (
            sum(
                particle.weight
                * np.outer(particle.pose - mean_pose, particle.pose - mean_pose)
                for particle in self.particles
            )
            / weight_sum
        )

        self.plot_best_particle.center = (mean_pose[0], mean_pose[1])
        (
            self.plot_best_particle.width,
            self.plot_best_particle.height,
            angle_rad,
        ) = self.get_ellipse_params(cov_pose)
        self.plot_best_particle.angle = angle_rad * 180.0 / pi

        # get trajectory points
        trajectory_x_list = [
            sum(
                particle.weight * particle.trajectory[i][0]
                for particle in self.particles
            )
            / weight_sum
            for i in range(0, len(self.particles[0].trajectory))
        ]
        trajectory_y_list = [
            sum(
                particle.weight * particle.trajectory[i][1]
                for particle in self.particles
            )
            / weight_sum
            for i in range(0, len(self.particles[0].trajectory))
        ]
        self.plot_trajectory.set_data(trajectory_x_list, trajectory_y_list)

        plots = [
            self.plot_landmarks,
            self.plot_particles,
            self.plot_best_particle,
            self.plot_trajectory,
        ]

        # draw the estimated landmark locations along with the ellipsoids
        for landmark in self.particles[index_of_best_particle].landmarks:
            if landmark.observed:
                bpx = landmark.mu[0]
                bpy = landmark.mu[1]
                plots.append(
                    self.ax.plot(bpx, bpy, "bo", markersize=3, animated=True)[0]
                )
                [a, b, angle] = self.get_ellipse_params(landmark.sigma)
                angle_degrees = angle * 180.0 / pi
                e = Ellipse([bpx, bpy], a, b, angle_degrees, fill=False, animated=True)
                self.ax.add_patch(e)
                plots.append(e)

        # draw the observations as lines between the best particle and the observed landmarks
        for measurement in sensor_measurements:
            landmark_x = (
                self.particles[index_of_best_particle]
                .landmarks[measurement.landmark_id]
                .mu[0]
            )
            landmark_y = (
                self.particles[index_of_best_particle]
                .landmarks[measurement.landmark_id]
                .mu[1]
            )
            plots.append(
                plt.plot(
                    (landmark_x, mean_pose[0]),
                    (landmark_y, mean_pose[1]),
                    "k",
                    linewidth=1,
                    animated=True,
                )[0]
            )

        return plots

    def get_ellipse_params(self, C):
        """Calculates unscaled half axes of the 95% covariance ellipse.

        C: covariance matrix
        alpha: confidence value

        Code from the CAS Robot Navigation Toolbox:
        http://svn.openslam.org/data/svn/cas-rnt/trunk/lib/drawprobellipse.m
        Copyright (C) 2004 CAS-KTH, ASL-EPFL, Kai Arras
        Licensed under the GNU General Public License, version 2
        """
        sxx = float(C[0, 0])
        syy = float(C[1, 1])
        sxy = float(C[0, 1])
        a = sqrt(
            0.5 * (sxx + syy + sqrt((sxx - syy) ** 2 + 4.0 * sxy ** 2))
        )  # always greater
        b = sqrt(
            0.5 * (sxx + syy - sqrt((sxx - syy) ** 2 + 4.0 * sxy ** 2))
        )  # always smaller

        # Remove imaginary parts in case of neg. definite C
        if not np.isreal(a):
            a = np.real(a)
        if not np.isreal(b):
            b = np.real(b)

        # Scaling in order to reflect specified probability
        a = a * sqrt(CHI_SQUARE_INV_95_2)
        b = b * sqrt(CHI_SQUARE_INV_95_2)

        # Look where the greater half axis belongs to
        if sxx < syy:
            swap = a
            a = b
            b = swap

        # Calculate inclination (numerically stable)
        if sxx != syy:
            angle = 0.5 * atan(2.0 * sxy / (sxx - syy))
        elif sxy == 0.0:
            angle = 0.0  # angle doesn't matter
        elif sxy > 0.0:
            angle = pi / 4.0
        elif sxy < 0.0:
            angle = -pi / 4.0
        return [a, b, angle]


def resample(particles):
    """Resample the set of particles.

    A particle has a probability proportional to its weight to get
    selected. A good option for such a resampling method is the so-called low
    variance sampling, Probabilistic Robotics page 109"""
    num_particles = len(particles)
    new_particles = []
    weights = [particle.weight for particle in particles]

    # normalize the weight
    sum_weights = sum(weights)
    weights = [weight / sum_weights for weight in weights]

    # the cumulative sum
    cumulative_weights = np.cumsum(weights)
    normalized_weights_sum = cumulative_weights[len(cumulative_weights) - 1]

    # check: the normalized weights sum should be 1 now (up to float representation errors)
    assert abs(normalized_weights_sum - 1.0) < 1e-5

    # initialize the step and the current position on the roulette wheel
    step = normalized_weights_sum / num_particles
    position = random.uniform(0, normalized_weights_sum)
    idx = 1

    # walk along the wheel to select the particles
    for i in range(1, num_particles + 1):
        position += step
        if position > normalized_weights_sum:
            position -= normalized_weights_sum
            idx = 1
        while position > cumulative_weights[idx - 1]:
            idx = idx + 1

        new_particles.append(copy.deepcopy(particles[idx - 1]))
        new_particles[i - 1].weight = 1 / num_particles

    return new_particles
