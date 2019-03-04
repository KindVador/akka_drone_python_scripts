# -*- coding: utf-8 -*-
import os
import argparse
import math
from pathlib import PurePath
from glob import glob
import pyulog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages


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

    modes_dict = {0: 'Manual', 1: 'Altitude control', 2: 'Position control', 3: 'Auto mission', 4: 'Auto loiter',
                  5: 'Auto return to launch', 6: 'RC recover', 7: 'Auto return to groundstation on data link loss',
                  8: 'Auto land on engine failure', 9: 'Auto land on gps failure', 10: 'Acro', 11: 'Free',
                  12: 'Descend', 13: 'Termination', 14: 'Offboard', 15: 'Stabilized', 16: 'Rattitude (aka "flip")',
                  17: 'Takeoff', 18: 'Land', 19: 'Auto Follow', 20: 'Precision land with landing target'}

    def __init__(self, ulog_file=None, start_time=None, end_time=None, lazy=True):
        super(PX4LogFile, self).__init__(ulog_file)
        self.data = {}
        if lazy:
            self.df = None
        else:
            self.df = self._create_dataframe()
        self.compute_additional_parameters()
        if start_time and end_time:
            self.df = self.df.between_time(start_time, end_time)

    def __getitem__(self, item):
        if self.df is not None and item in self.df.columns:
            return self.df[item]
        else:
            self._add_parameter_to_df(item)
            return self.df[item]

    def _add_parameter_to_df(self, var_name):
        if '.' in var_name:
            msg_name = var_name.split('.')[0]
            parameter_name = var_name.split('.')[1]
            ds = self.get_dataset(msg_name)
            ts = ds.data['timestamp']
            if self.df is None:
                self.df = pd.DataFrame(ds.data[parameter_name], index=pd.to_datetime(ts, unit='us'), columns=[var_name])
            else:
                df2 = pd.DataFrame(ds.data[parameter_name], index=pd.to_datetime(ts, unit='us'), columns=[var_name])
                try:
                    if self.df.shape[0] > df2.shape[0]:
                        self.df = pd.concat([self.df, df2], axis=1)
                    else:
                        self.df = pd.concat([df2, self.df], axis=1)
                except ValueError as ve:
                    print(f'ERROR for variable: {var_name}')
                    print(ve)
        else:
            raise ValueError("Invalid parameter, it should be composed of message_name.parameter_name ")

        self.df.fillna(method='ffill', inplace=True)

    def _create_dataframe(self):
        df = None
        for msg in self.data_list:
            ds = self.get_dataset(msg.name)
            ts = ds.data['timestamp']
            for k, v in msg.data.items():
                var_name = f'{msg.name}.{k}'
                if k == 'timestamp' or (df is not None and var_name in df.columns):
                    continue
                if df is None:
                    df = pd.DataFrame(ds.data[k], index=pd.to_datetime(ts, unit='us'), columns=[var_name])
                else:
                    df2 = pd.DataFrame(ds.data[k], index=pd.to_datetime(ts, unit='us'), columns=[var_name])
                    try:
                        if df.shape[0] > df2.shape[0]:
                            df = pd.concat([df, df2], axis=1)
                        else:
                            df = pd.concat([df2, df], axis=1)
                    except ValueError as ve:
                        print(f'ERROR for variable: {var_name}')
                        print(ve)
        df.fillna(method='ffill', inplace=True)
        return df

    def compute_additional_parameters(self):
        # motors 2 & 3
        self.df['left_motors'] = (self['actuator_outputs.output[1]'] + self['actuator_outputs.output[2]']) / 2
        # motors 1 & 4
        self.df['right_motors'] = (self['actuator_outputs.output[0]'] + self['actuator_outputs.output[3]']) / 2
        # compute thrust
        self.df['thrust'] = (self['left_motors'] + self['right_motors']) / 2
        # compute euler angles from quaternion
        va = self.get_dataset('vehicle_attitude')
        time_data = va.data['timestamp']
        for i in range(4):
            _ = self[f'vehicle_attitude.q[{str(i)}]']
        roll, pitch, yaw = vquaternion2euler(va.data['q[0]'], va.data['q[1]'], va.data['q[2]'], va.data['q[3]'])
        # add time data to each angle and add to parameters list
        self.df['roll'] = pd.Series(roll, index=pd.to_datetime(time_data, unit='us'), name='roll')
        self.df['pitch'] = pd.Series(pitch, index=pd.to_datetime(time_data, unit='us'), name='pitch')
        self.df['yaw'] = pd.Series(yaw, index=pd.to_datetime(time_data, unit='us'), name='yaw')
        # compute lift force
        self.df['lift'] = self['thrust'] * np.cos(self['roll']) * np.cos(self['pitch'])
        self.df.fillna(method='ffill', inplace=True)

    def create_plots(self, message_name, parameters, pdf_file, control_modes=None):

        fig, axs = plt.subplots(len(parameters), 1, sharex='all')
        for i in range(len(parameters)):
            axs[i].plot(self[f'{message_name}.{parameters[i]}'], drawstyle='steps-post')
            axs[i].set_ylabel(parameters[i])
            axs[i].grid(True)
        if control_modes:
            for a in range(len(axs)):
                for k, v in control_modes.items():
                    axs[a].axvline(x=k, color='b', linestyle='--', label=v)
                    if a == 0:
                        y_pos = axs[a].get_yaxis().axes.get_ylim()
                        axs[0].text(k, y_pos[1], v, rotation=90, ha='right', va='bottom')
        fig.tight_layout()
        pdf_file.savefig(fig, dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype='a4',
                         transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)

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

    def create_lateral_plots(self, pdf_file, control_modes=None):
        fig, axs = plt.subplots(4, 1, sharex='all')
        axs[0].set_title('Lateral axis')
        axs[0].plot(self['roll'] * __rad2deg__, drawstyle='steps-post')
        axs[0].set_ylabel('Roll')
        axs[0].grid(True)
        axs[1].plot(self['left_motors'], drawstyle='steps-post')
        axs[1].set_ylabel('Left motors')
        axs[1].grid(True)
        axs[2].plot(self['right_motors'], drawstyle='steps-post')
        axs[2].set_ylabel('Right motors')
        axs[2].grid(True)
        axs[3].plot(self['left_motors'] - self['right_motors'], drawstyle='steps-post')
        axs[3].grid(True)
        if control_modes:
            for a in range(len(axs)):
                for k, v in control_modes.items():
                    axs[a].axvline(x=k, color='b', linestyle='--', label=v)
                    if a == 0:
                        y_pos = axs[a].get_yaxis().axes.get_ylim()
                        axs[0].text(k, y_pos[1], v, rotation=90, ha='right', va='bottom')
        fig.tight_layout()
        pdf_file.savefig(fig, dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype='a4',
                         transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)

    def create_attitude_plots(self, pdf_file, control_modes=None):
        fig, axs = plt.subplots(3, 1, sharex='all')
        axs[0].set_title('vehicle_attitude')
        axs[0].plot(self['roll'] * __rad2deg__, drawstyle='steps-post')
        axs[0].set_ylabel('Roll')
        axs[0].grid(True)
        axs[1].plot(self['pitch'] * __rad2deg__, drawstyle='steps-post')
        axs[1].set_ylabel('Pitch')
        axs[1].grid(True)
        axs[2].plot(self['yaw'] * __rad2deg__, drawstyle='steps-post')
        axs[2].set_ylabel('Yaw')
        axs[2].grid(True)
        if control_modes:
            for a in range(len(axs)):
                for k, v in control_modes.items():
                    axs[a].axvline(x=k, color='b', linestyle='--', label=v)
                    if a == 0:
                        y_pos = axs[a].get_yaxis().axes.get_ylim()
                        axs[0].text(k, y_pos[1], v, rotation=90, ha='right', va='bottom')
        fig.tight_layout()
        fig.tight_layout()
        pdf_file.savefig(fig, dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype='a4',
                         transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)

    def create_vertical_plots(self, pdf_file, control_modes=None):
        fig, axs = plt.subplots(3, 1, sharex='all')
        axs[0].set_title('Vertical axis')
        axs[0].plot(self['thrust'], drawstyle='steps-post')
        axs[0].plot(self['lift'], drawstyle='steps-post')
        axs[0].set_ylabel('Thrust & Lift')
        axs[0].grid(True)
        axs[1].plot(self['vehicle_global_position.alt'], drawstyle='steps-post')
        axs[1].set_ylabel('Altitude')
        axs[1].grid(True)
        if control_modes:
            for a in range(len(axs)):
                for k, v in control_modes.items():
                    axs[a].axvline(x=k, color='b', linestyle='--', label=v)
                    if a == 0:
                        y_pos = axs[a].get_yaxis().axes.get_ylim()
                        axs[0].text(k, y_pos[1], v, rotation=90, ha='right', va='bottom')
        # vehicle_status.nav_state
        axs[2].plot(self['vehicle_status.nav_state'], drawstyle='steps-post')
        axs[2].set_ylabel('Navigation State')
        axs[2].grid(True)
        fig.tight_layout()
        pdf_file.savefig(fig, dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype='a4',
                         transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)


def main(input, start_time=None, end_time=None):
    if os.path.isdir(input):
        logs = glob(f'{input}/**/*.ulg', recursive=True)
    else:
        logs = [input]

    for log in logs:
        p = PurePath(log)

        # create folder
        folder_abspath = os.path.join(p.parents[0], p.stem)
        if not os.path.exists(folder_abspath):
            os.mkdir(folder_abspath)

        log = PX4LogFile(log, start_time=start_time, end_time=end_time)

        # print in console some general information
        print('Start timestamp: ', log.df.index[0].strftime('%Y-%m-%d %H:%M:%S'))
        print('Last timestamp: ', log.df.index[-1].strftime('%Y-%m-%d %H:%M:%S'))

        # find timestamps for each control mode
        col = 'vehicle_status.nav_state'
        control_modes = {}
        for idx in log[col].diff()[log[col].diff() != 0].index.values:
            try:
                control_modes[idx] = PX4LogFile.modes_dict[log.df.at[idx, col]]
            except KeyError as ke:
                pass

        # Create PDF file
        pdf_file = PdfPages(os.path.join(folder_abspath, f'{p.stem}.pdf'))

        log.create_plots('vehicle_global_position', ['alt', 'pressure_alt', 'terrain_alt'], pdf_file, control_modes=control_modes)
        log.create_plots('vehicle_command', ['param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'param7'], pdf_file, control_modes=control_modes)
        log.create_plots('vehicle_attitude', ['rollspeed', 'pitchspeed', 'yawspeed'], pdf_file, control_modes=control_modes)
        log.create_plots('actuator_outputs', ['output[0]', 'output[1]', 'output[2]', 'output[3]', 'output[4]', 'output[5]'], pdf_file, control_modes=control_modes)

        # list of parameters recorded in the log
        # log.print_available_parameters()

        log.create_attitude_plots(pdf_file, control_modes=control_modes)
        log.create_lateral_plots(pdf_file, control_modes=control_modes)
        log.create_vertical_plots(pdf_file, control_modes=control_modes)

        pdf_file.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", help="path to the log file (*.ulg) or folder with log files")
    ap.add_argument("-s", "--start", help="start timestamp (e.g. 00:05:10)")
    ap.add_argument("-e", "--end", help="end timestamp (e.g. 00:05:30)")
    args = vars(ap.parse_args())
    if args['input'] and not os.path.isdir(args["input"]):
        if args['start'] and args['end']:
            main(input=args["input"], start_time=args['start'], end_time=args['end'])
        else:
            main(input=args["input"])
    elif args['input'] and os.path.isdir(args["input"]):
        if 'start' in args and 'end' in args:
            print('start and end arguments are ignored when a folder is passed as input')
        main(input=args["input"])
    else:
        raise ValueError('Missing argument')
