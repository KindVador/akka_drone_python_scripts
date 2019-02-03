# -*- coding: utf-8 -*-
import os
import argparse
import math
from pathlib import Path, PurePath
import pyulog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams


rcParams['figure.figsize'] = (20, 10)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
rcParams['figure.subplot.top'] = 0.977
rcParams['figure.subplot.bottom'] = 0.073
rcParams['figure.subplot.left'] = 0.036
rcParams['figure.subplot.right'] = 0.99
rcParams['figure.subplot.hspace'] = 0.2
rcParams['figure.subplot.wspace'] = 0.2

plt.style.use('ggplot')


__rad2deg__ = 180.0 / math.pi


def quaternion2euler(q0, q1, q2, q3):
    """
    Quaternion rotation from NED earth frame to XYZ body frame

    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    Args:
        q0 (float): first quaternion.
        q1 (float): second quaternion.
        q2 (float): third quaternion.
        q3 (float): fourth quaternion.

    Returns:
        tuple: euler angles in radians (roll, pitch, yaw)
    """

    roll = math.atan2(2*(q0*q1 + q2*q3), 1 - 2*(q1*q1 + q2*q2))
    pitch = math.asin(2*(q0*q2 - q3*q1))
    yaw = math.atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2*q2 + q3*q3))
    return roll, pitch, yaw


# create a vectorize function
vquaternion2euler = np.vectorize(quaternion2euler, otypes=[np.float, np.float, np.float])


class PX4LogFile(pyulog.ULog):

    def __init__(self, ulog_file=None):
        super(PX4LogFile, self).__init__(ulog_file)
        self.data = {}
        self.compute_additional_parameters()

    def create_dataframe(self):
        pass

    def compute_additional_parameters(self):
        # compute euler angles from quaternion
        va = self.get_dataset('vehicle_attitude')
        time_data = va.data['timestamp']
        roll, pitch, yaw = vquaternion2euler(va.data['q[0]'], va.data['q[1]'], va.data['q[2]'], va.data['q[3]'])
        # add time data to each angle and add to parameters list
        self.data['roll'] = np.stack([time_data, roll], axis=1)
        self.data['pitch'] = np.stack([time_data, pitch], axis=1)
        self.data['yaw'] = np.stack([time_data, yaw], axis=1)

        ao = self.get_dataset('actuator_outputs')
        # motors 2 & 3
        left_motors = (ao.data['output[1]'] + ao.data['output[2]']) / 2
        self.data['left_motors'] = np.stack([ao.data['timestamp'], left_motors], axis=1)
        # motors 1 & 4
        right_motors = (ao.data['output[0]'] + ao.data['output[3]']) / 2
        self.data['right_motors'] = np.stack([ao.data['timestamp'], right_motors], axis=1)
        thrust = (left_motors + right_motors) / 4
        self.data['thrust'] = np.stack([ao.data['timestamp'], thrust], axis=1)

        # compute lift force
        # lift = thrust * np.cos(vroll) * np.cos(vpitch)

    def create_plots(self, message_name, parameters, file_name=None, abs_path=None):

        msg = self.get_dataset(message_name)
        time_data = msg.data['timestamp']
        fig, axs = plt.subplots(len(parameters), 1, sharex='all')
        for i in range(len(parameters)):
            axs[i].plot(time_data, msg.data[parameters[i]], drawstyle='steps-post')
            axs[i].set_ylabel(parameters[i])
            axs[i].grid(True)
        fig.tight_layout()
        if file_name is None:
            file_name = message_name
        if abs_path is None:
            abs_path = Path.cwd()
        path_file = os.path.join(abs_path, f"{file_name}.pdf")

        fig.savefig(path_file, dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype='a4',
                    format='pdf', transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)

    def available_parameters(self):
        _ap = []
        for msg in self.data_list:
            for k, v in msg.data.items():
                _ap.append('{}.{}'.format(msg.name, k))

        return _ap

    def print_initial_parameters(self):

        for k, v in self.initial_parameters.items():
            print(k, v)

    def print_changed_parameters(self):

        for cp in self.changed_parameters:
            print(cp)

    def print_message_formats(self):

        for k, v in self.message_formats.items():
            print("{} ({}):".format(v.name, dir(v)))
            for f in v.fields:
                print("\t\t{}".format(f))

    def print_logged_messages(self):

        for lm in self.logged_messages:
            print(lm)

    def print_dropouts(self):

        for d in self.dropouts:
            print(d)

    def print_available_parameters(self):

        for msg in self.data_list:
            for k, v in msg.data.items():
                print('{}.{}'.format(msg.name, k))

    def create_lateral_plots(self, abs_path=None):
        fig, axs = plt.subplots(4, 1, sharex='all')
        axs[0].set_title('Lateral axis')
        axs[0].plot(self.data['roll'][:, 0], self.data['roll'][:, 1] * __rad2deg__, drawstyle='steps-post')
        axs[0].set_ylabel('Roll')
        axs[0].grid(True)
        axs[1].plot(self.data['left_motors'][:, 0], self.data['left_motors'][:, 1], drawstyle='steps-post')
        axs[1].set_ylabel('Left motors')
        axs[1].grid(True)
        axs[2].plot(self.data['right_motors'][:, 0], self.data['right_motors'][:, 1], drawstyle='steps-post')
        axs[2].set_ylabel('Right motors')
        axs[2].grid(True)
        axs[3].plot(self.data['right_motors'][:, 0], self.data['left_motors'][:, 1] - self.data['right_motors'][:, 1], drawstyle='steps-post')
        axs[3].grid(True)
        fig.tight_layout()
        if abs_path is None:
            abs_path = Path.cwd()
        path_file = os.path.join(abs_path, "lateral_axis.pdf")
        fig.savefig(path_file, dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype='a4',
                    format='pdf', transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)

    def create_attitude_plots(self, abs_path=None):
        fig, axs = plt.subplots(3, 1, sharex='all')
        axs[0].set_title('vehicle_attitude')
        axs[0].plot(self.data['roll'][:, 0], self.data['roll'][:, 1] * __rad2deg__, drawstyle='steps-post')
        axs[0].set_ylabel('Roll')
        axs[0].grid(True)
        axs[1].plot(self.data['pitch'][:, 0], self.data['pitch'][:, 1] * __rad2deg__, drawstyle='steps-post')
        axs[1].set_ylabel('Pitch')
        axs[1].grid(True)
        axs[2].plot(self.data['yaw'][:, 0], self.data['yaw'][:, 1] * __rad2deg__, drawstyle='steps-post')
        axs[2].set_ylabel('Yaw')
        axs[2].grid(True)
        fig.tight_layout()
        if abs_path is None:
            abs_path = Path.cwd()
        path_file = os.path.join(abs_path, "vehicule_attitude.pdf")
        fig.savefig(path_file, dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
                    papertype='a4',
                    format='pdf', transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)

    def create_vertical_plots(self, abs_path=None):
        fig, axs = plt.subplots(2, 1, sharex='all')
        axs[0].set_title('Vertical axis')
        axs[0].plot(self.data['thrust'][:, 0], self.data['thrust'][:, 1], drawstyle='steps-post')
        # axs[0].plot(time_data_outputs, lift, drawstyle='steps-post')
        axs[0].set_ylabel('Thrust & Lift')
        axs[0].grid(True)
        vgp = self.get_dataset('vehicle_global_position')
        time_global_position = vgp.data['timestamp']
        alt = vgp.data['alt']
        axs[1].plot(time_global_position, alt, drawstyle='steps-post')
        axs[1].set_ylabel('Altitude')
        axs[1].grid(True)
        fig.tight_layout()
        if abs_path is None:
            abs_path = Path.cwd()
        path_file = os.path.join(abs_path, "vertical_axis.pdf")
        fig.savefig(path_file, dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
                    papertype='a4',
                    format='pdf', transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)


def main(ulog_file):

    # create folder
    p = PurePath(ulog_file)
    folder_abspath = os.path.join(p.parents[0], p.stem)
    if not os.path.exists(folder_abspath):
        os.mkdir(folder_abspath)

    log = PX4LogFile(ulog_file)

    log.create_plots('vehicle_global_position', ['alt', 'pressure_alt', 'terrain_alt'], abs_path=folder_abspath)
    log.create_plots('vehicle_command', ['param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'param7'], abs_path=folder_abspath)
    log.create_plots('vehicle_attitude', ['rollspeed', 'pitchspeed', 'yawspeed'], file_name='vehicle_attitude_rates', abs_path=folder_abspath)
    log.create_plots('actuator_outputs', ['output[0]', 'output[1]', 'output[2]', 'output[3]', 'output[4]', 'output[5]'], abs_path=folder_abspath)

    # list of parameters recorded in the log
    log.print_available_parameters()

    log.create_attitude_plots(abs_path=folder_abspath)
    log.create_lateral_plots(abs_path=folder_abspath)
    log.create_vertical_plots(abs_path=folder_abspath)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", help="path to the log file (*.ulg)")
    ap.add_argument("-s", "--start", help="start timestamp")
    ap.add_argument("-e", "--end", help="end timestamp")
    args = vars(ap.parse_args())
    main(args["file"])
