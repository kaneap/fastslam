class Odometry(object):
    """Represents an odometry command.

    - r1: initial rotation in radians counterclockwise
    - t: translation in meters
    - r2: final rotation in radians counterclockwise
    """

    def __init__(self, r1, t, r2):
        self.r1 = r1
        self.t = t
        self.r2 = r2

    def __str__(self):
        return "Odometry(r1 = {0} rad, t = {1} m, r2 = {2} rad)".format(
            self.r1, self.t, self.r2
        )
