# Copyright (C) 2025 Edward Morgan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.

import numpy as np

from simlab.uvms_parameters_sim import (
    ReachParams as SimReachParams,
    VehicleControllerParams as SimVehicleControllerParams,
)


class ReachParams(SimReachParams):
    profile_name = "real"

    pid_kp = np.array([2.0, 3.0, 1.0, 1.5])
    pid_ki = np.array([0.0, 0.0, 0.0, 0.0])
    pid_kd = np.array([0.0, 0.0, 0.0, 0.0])

class VehicleControllerParams(SimVehicleControllerParams):
    profile_name = "real"
    pid_kp = np.array([40.0, 40.0, 100.0, 2.0, 2.0, 10.0], dtype=float)
    pid_kd = np.array([15.0, 15.0, 30.0, 2.0, 2.0, 5.0], dtype=float)
    pid_ki = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 3.0], dtype=float)