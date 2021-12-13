class SensorMeasurement(object):
    """Represents one or more range and bearing sensor measurements.

    - id: ID of the landmark
    - z_range: measured distance to the landmark in meters
    - z_bearing: measured angle towards the landmark in radians
    """

    def __init__(self, landmark_id, z_range, z_bearing):
        self.landmark_id = landmark_id
        self.z_range = z_range
        self.z_bearing = z_bearing

    def __str__(self):
        return "SensorMeasurement(landmark_id = {0}, z_range = {1} m, z_bearing = {2} rad)".format(
            self.landmark_id, self.z_range, self.z_bearing
        )
