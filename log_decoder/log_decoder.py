# -*- coding: utf-8 -*-
import os
import argparse
import math
import pyulog
import numpy as np
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


def create_plots(log_file, message_name, parameters, file_name=None):
    if isinstance(log_file, pyulog.ULog):
        ulog = log_file
    elif isinstance(log_file, str):
        if os.path.isfile(log_file):
            ulog = pyulog.ULog(log_file)
        else:
            raise FileNotFoundError(log_file)

    msg = ulog.get_dataset(message_name)
    time_data = msg.data['timestamp']
    fig, axs = plt.subplots(len(parameters), 1, sharex='all')
    for i in range(len(parameters)):
        axs[i].plot(time_data, msg.data[parameters[i]], drawstyle='steps-post')
        axs[i].set_ylabel(parameters[i])
        axs[i].grid(True)
    fig.tight_layout()
    if file_name is None:
        file_name = message_name
    fig.savefig(f"{file_name}.pdf", dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype='a4',
                format='pdf', transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)


def print_initial_parameters(log_file):
    if isinstance(log_file, pyulog.ULog):
        ulog = log_file
    elif isinstance(log_file, str):
        if os.path.isfile(log_file):
            ulog = pyulog.ULog(log_file)
        else:
            raise FileNotFoundError(log_file)

    for k, v in ulog.initial_parameters.items():
        print(k, v)


def print_changed_parameters(log_file):
    if isinstance(log_file, pyulog.ULog):
        ulog = log_file
    elif isinstance(log_file, str):
        if os.path.isfile(log_file):
            ulog = pyulog.ULog(log_file)
        else:
            raise FileNotFoundError(log_file)

    for cp in ulog.changed_parameters:
        print(cp)


def print_message_formats(log_file):
    if isinstance(log_file, pyulog.ULog):
        ulog = log_file
    elif isinstance(log_file, str):
        if os.path.isfile(log_file):
            ulog = pyulog.ULog(log_file)
        else:
            raise FileNotFoundError(log_file)

    for k, v in ulog.message_formats.items():
        print("{} ({}):".format(v.name, dir(v)))
        for f in v.fields:
            print("\t\t{}".format(f))


def print_logged_messages(log_file):
    if isinstance(log_file, pyulog.ULog):
        ulog = log_file
    elif isinstance(log_file, str):
        if os.path.isfile(log_file):
            ulog = pyulog.ULog(log_file)
        else:
            raise FileNotFoundError(log_file)

    for lm in ulog.logged_messages:
        print(lm)


def print_dropouts(log_file):
    if isinstance(log_file, pyulog.ULog):
        ulog = log_file
    elif isinstance(log_file, str):
        if os.path.isfile(log_file):
            ulog = pyulog.ULog(log_file)
        else:
            raise FileNotFoundError(log_file)

    for d in ulog.dropouts:
        print(d)


def print_available_parameters(log_file):
    if isinstance(log_file, pyulog.ULog):
        ulog = log_file
    elif isinstance(log_file, str):
        if os.path.isfile(log_file):
            ulog = pyulog.ULog(log_file)
        else:
            raise FileNotFoundError(log_file)

    for msg in ulog.data_list:
        for k, v in msg.data.items():
            print('{}.{}'.format(msg.name, k))


def main(ulog_file):

    log_file = pyulog.ULog(ulog_file)

    create_plots(log_file, 'vehicle_global_position', ['alt', 'pressure_alt', 'terrain_alt'])
    create_plots(log_file, 'vehicle_command', ['param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'param7'])
    create_plots(log_file, 'vehicle_attitude', ['rollspeed', 'pitchspeed', 'yawspeed'], file_name='vehicle_attitude_rates')
    create_plots(log_file, 'actuator_outputs', ['output[0]', 'output[1]', 'output[2]', 'output[3]', 'output[4]', 'output[5]'])

    # list of parameters recorded in the log
    print_available_parameters(log_file)

    # vehicle_attitude
    vehicle_attitude = log_file.get_dataset('vehicle_attitude')
    time_data = vehicle_attitude.data['timestamp']
    q0 = vehicle_attitude.data['q[0]']
    q1 = vehicle_attitude.data['q[1]']
    q2 = vehicle_attitude.data['q[2]']
    q3 = vehicle_attitude.data['q[3]']
    quaternion2euler_array = np.frompyfunc(quaternion2euler, 4, 3)
    roll, pitch, yaw = quaternion2euler_array(q0, q1, q2, q3)
    fig, axs = plt.subplots(3, 1, sharex='all')
    axs[0].set_title('vehicle_attitude')
    axs[0].plot(time_data, roll * __rad2deg__, drawstyle='steps-post')
    axs[0].set_ylabel('Roll')
    axs[0].grid(True)
    axs[1].plot(time_data, pitch * __rad2deg__, drawstyle='steps-post')
    axs[1].set_ylabel('Pitch')
    axs[1].grid(True)
    axs[2].plot(time_data, yaw * __rad2deg__, drawstyle='steps-post')
    axs[2].set_ylabel('Yaw')
    axs[2].grid(True)
    fig.tight_layout()
    fig.savefig(f"vehicle_attitude.pdf", dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype='a4',
                format='pdf', transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)

    # LATERAL AXE
    actuator_outputs = log_file.get_dataset('actuator_outputs')
    time_data_outputs = actuator_outputs.data['timestamp']
    # motors 2 & 3
    left_motors = (actuator_outputs.data['output[1]'] + actuator_outputs.data['output[2]']) / 2
    # motors 1 & 4
    right_motors = (actuator_outputs.data['output[0]'] + actuator_outputs.data['output[3]']) / 2

    thrust = (actuator_outputs.data['output[1]'] + actuator_outputs.data['output[2]'] + actuator_outputs.data['output[0]'] + actuator_outputs.data['output[3]']) / 4
    fig, axs = plt.subplots(4, 1, sharex='all')
    axs[0].plot(time_data, roll * __rad2deg__, drawstyle='steps-post')
    axs[0].set_ylabel('Roll')
    axs[0].grid(True)
    axs[1].plot(time_data_outputs, left_motors, drawstyle='steps-post')
    axs[1].set_ylabel('Left motors')
    axs[1].grid(True)
    axs[2].plot(time_data_outputs, right_motors, drawstyle='steps-post')
    axs[2].set_ylabel('Right motors')
    axs[2].grid(True)
    axs[3].plot(time_data_outputs, left_motors - right_motors, drawstyle='steps-post')
    axs[3].grid(True)
    fig.tight_layout()
    fig.savefig(f"axe_lateral.pdf", dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype='a4',
                format='pdf', transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", help="path to the log file (*.ulg)")
    args = vars(ap.parse_args())
    main(args["file"])
