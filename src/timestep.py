class TimeStep(object):
    """Represents a data point consisting of an odometry command and a list of sensor measurements.

    - odom: Odometry measurement
    - sensor: List of landmark observations of type SensorMeasurement
    """

    def __init__(self, odom, sensor):
        self.odom = odom
        self.sensor = sensor

    def __str__(self):
        return "TimeStep(odom = {0}, sensor = {1})".format(self.odom, self.sensor)