import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from utils import *
import screen_capture_rev2 as scr2
import downsampling as ds

import matplotlib as mpl
mpl.rc('font', family='Times New Roman', size=18)

colors = ['r', 'g', 'b', 'c', 'm', 'y']
    
#function for the guassian pdf
def normal(x, mean=0, stdev=1):
    return (1 / (stdev * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mean) / stdev)**2)

#function for finite derivative    
def calc_time_deriv(time, data):
    n_pts, n_dims = np.shape(data)
    deriv = np.zeros((n_pts-1, n_dims))
    for i in range(n_pts-1):
        deriv[i] = (data[i + 1] - data[i]) / (time[i + 1] - time[i])
    return deriv

#function for calculating jerk using finite derivatives
def calc_jerk_in_time(time, position):
    data = position
    for _ in range(3):
        data = calc_time_deriv(time, data)
    return data

#function to calculate moving average
# data: data for moving average
# window_size: the window size of moving average (number of data points to average together)
def moving_average(data, window_size):
    avg_data = []
    for i in range(len(data) - window_size):
        avg_data.append(np.mean(data[i:i+window_size]))
    return np.array(avg_data)

#function for thresholding a signal to find changepoints
# data: data to threshold
# threshold: the threshold over which data must reach in order to be considered a changepoint
# window_size: the data must stay above the threshold for at least this many samples to be considered a segment
# grace threshold: if data dips below the threshold for less samples than the grace_threshold, it is not considered a new changepoint (this is to reduce error from noisy signals)
def count_thresh(data, threshold, segment_size, grace_threshold):
    segments = [0]
    count = 0
    grace_count = 0
    for i in range(len(data)):
        if data[i] >= threshold:
            if count >= segment_size:
                if segments[-1] + count != i:
                    segments.append(i - count)
            count = count + 1
            grace_count = 0
        else:
            if grace_count > grace_threshold:
                count = 0
            else:
                grace_count = grace_count + 1
                count = count + 1
    print(segments)
    return segments

#function to segment a given data stream
#Given a data stream, returns segments for the given data stream. First the jerk of that data stream is calculated, then a moving average is applied to the jerk signal. This averaged signal is then thresholded to find changepoints.
# time: time data for the given data stream
# data: data stream to segment
# base_thresh: the threshold for changepoint detection
# segment_size: the segment size for changepoint detection
# window_size: the window size for moving average
# grace_thresh: the grace threshold for changepoint detection
# plot: whether or not to plot results
def segment(time, data, base_thresh=1000, segment_size=256, window_size=64, grace_thresh=32, plot=False):
    jerk = calc_jerk_in_time(time, data)
    total_jerk = np.linalg.norm(jerk, axis=1)
    avg_jerk = moving_average(total_jerk, window_size)
    norm_avg_jerk = avg_jerk / np.max(avg_jerk)
    segments = count_thresh(norm_avg_jerk, base_thresh, segment_size, grace_thresh)
    for i in range(1, len(segments)):
        segments[i] = segments[i] + window_size // 2
    if plot:
        fig = plt.figure(figsize=(7, 6))
        plt.title('Changepoint Detection')
        color_ind = 0
        for i in range(len(norm_avg_jerk)):
            plt.plot(i, norm_avg_jerk[i], colors[color_ind % len(colors)] + '.', ms=12)
            if color_ind < len(segments):
                if (segments[color_ind] == i):
                    color_ind = color_ind + 1
        plt.xlabel('Time')
        plt.ylabel('Jerk')
        plt.show()
    return segments

#function to probabilistically model changepoints
#Given a list of changepoints, returns a list of probabilities corresponding to the changepoints. This is done by modelling a gaussian around each changepoint.
# segment_list: list of segment points
# data_len: total length of the data (need this to know how many probability values to return)
# window_size: the window size used for the moving average, determines the standard deviation of the modelled gaussian
# plot: whether or not to plot results
def calc_segment_prob(segment_list, data_len, window_size, plot=False):
    probabilities = np.ones((data_len,))
    for segment_point in segment_list:
        probabilities = probabilities + normal(np.linspace(0, 1, data_len), mean=segment_point / data_len, stdev=window_size / data_len)
    probabilities = probabilities / np.sum(probabilities)
    if plot:
        fig = plt.figure(figsize=(7, 6))
        plt.title('Segment ' + str(segment_list) + ' Probabilities')
        plt.plot(probabilities, lw=5)
        plt.xlabel('Time')
        plt.ylabel('Keypoint Probability')
        plt.show()
    return probabilities

#function to probabilistically model changepoints for multiple lists of changepoints
#Given a list of lists of changepoints, returns a list of probabilities corresponding to the combined changepoints. This is done by finding probabilities for each list of changepoints, then multiplying all probabilities together to find a probability over all changepoint lists.
# list_of_list_of_segments: list of lists of segment points
# data_len: total length of the data (need this to know how many probability values to return)
# window_size: the window size used for the moving average, determines the standard deviation of the modelled gaussian
# plot: whether or not to plot results
def calc_prob_from_segments(list_of_list_of_segments, data_len, window_size, plot=False):
    probabilities = np.ones((data_len,))
    for segment_list in list_of_list_of_segments:
        segment_probabilities = calc_segment_prob(segment_list[1:], data_len, window_size, plot)
        probabilities = probabilities * segment_probabilities
    probabilities = probabilities / np.sum(probabilities)
    if plot:
        plt.figure()
        plt.title('Combined Probabilities')
        plt.plot(probabilities)
        plt.show()
    return probabilities

#function to probabilistically model changepoints and sample keypoints
#Given a list of lists of changepoints, returns a list of keypoints over all data streams. This is done by first probabilistically modelling the changepoints, combining the probabilistic models, and sampling from the calculated probabilities.
# list_of_list_of_segments: list of lists of segment points
# data_len: total length of the data (need this to know how many probability values to return)
# window_size: the window size used for the moving average, determines the standard deviation of the modelled gaussian and if closely sampled keypoints can be combined into a single keypoint
# n_samples: how many times to sample the combined probabilities
# n_pass: how many samples within a given window are needed in order to be considered a keypoint
# plot: whether or not to plot results of probabilistic modelling
def probabilistically_combine(list_of_list_of_segments, data_len, window_size, n_samples=10, n_pass=2, plot=False):
    probabilities = calc_prob_from_segments(list_of_list_of_segments, data_len, window_size, plot)
    keypoints = np.random.choice(data_len, size=n_samples, replace=True, p=probabilities)
    print('Chosen Keypoints')
    print(keypoints)
    sorted_keys = np.sort(keypoints)
    final_keys = []
    cur_key = 0
    cur_key_group = []
    for i in range(len(sorted_keys)):
        if sorted_keys[i] <= cur_key + window_size:
            cur_key_group.append(sorted_keys[i])
        else:
            if len(cur_key_group) >= n_pass:
                final_keys.append(int(np.mean(cur_key_group)))
            cur_key = sorted_keys[i]
            cur_key_group = [cur_key]
    if len(cur_key_group) >= n_pass:
        final_keys.append(int(np.mean(cur_key_group)))
    print('Unique Sorted Keypoints')
    print(final_keys)
    keypoints = np.insert(final_keys, 0, 0)
    keypoints = np.append(keypoints, data_len)
    return keypoints

#function to perform full process of changepoint detection, probabilistic modelling, and keypoint sampling
#See above functions for description of parameters.
def full_segmentation(time, list_of_data, base_thresh=1000, segment_size=256, window_size=64, grace_thresh=32, n_samples=10, n_pass=2, plot=False):
    list_of_segments = [segment(time, data, base_thresh=base_thresh, segment_size=segment_size, window_size=window_size, grace_thresh=grace_thresh, plot=plot) for data in list_of_data]
    segments = probabilistically_combine(list_of_segments, len(time), window_size, n_samples=n_samples, n_pass=n_pass, plot=plot)
    return segments

#Example process using a 3D trajectory with multiple data streams
def main3d():
    seed = 440773
    np.random.seed(seed)
    fname = '../h5 files/three_button_pressing_demo.h5'
    joint_data, tf_data, wrench_data, gripper_data = read_robot_data(fname)
    
    joint_time = joint_data[0][:, 0] + joint_data[0][:, 1] * (10.0**-9)
    joint_pos = np.unwrap(joint_data[1], axis=0)
    
    traj_time = tf_data[0][:, 0] + tf_data[0][:, 1] * (10.0**-9)
    traj_pos = tf_data[1]
    
    wrench_time = wrench_data[0][:, 0] + wrench_data[0][:, 1] * (10.0**-9)
    wrench_frc = wrench_data[1]
    
    gripper_time = gripper_data[0][:, 0] + gripper_data[0][:, 1] * (10.0**-9)
    gripper_pos = gripper_data[1]
    
    traj_pos, ds_inds = ds.DouglasPeuckerPoints2(traj_pos, 1000)
    
    
    joint_time = joint_time[ds_inds]
    joint_pos = joint_pos[ds_inds, :]
    traj_time = traj_time[ds_inds]
    wrench_time = wrench_time[ds_inds]
    wrench_frc = wrench_frc[ds_inds, :]
    gripper_time = gripper_time[ds_inds]
    gripper_pos = gripper_pos[ds_inds]
    
    print('Joint Positions')
    thresh = 0.2
    ssize = 64
    wsize = 64
    gthresh = 4
    joint_segments = segment(joint_time, joint_pos, base_thresh=thresh, segment_size=ssize, window_size=wsize, grace_thresh=gthresh, plot=False)
    
    print('Trajectory')
    thresh = 0.25
    ssize = 64
    wsize = 64
    gthresh = 4
    traj_segments = segment(traj_time, traj_pos, base_thresh=thresh, segment_size=ssize, window_size=wsize, grace_thresh=gthresh, plot=False)
    
    print('Wrench Force')
    thresh = 0.15
    ssize = 64
    wsize = 64
    gthresh = 4
    frc_segments = segment(wrench_time, wrench_frc, base_thresh=thresh, segment_size=ssize, window_size=wsize, grace_thresh=gthresh, plot=False)
    
    print('Gripper')
    thresh = 0.1
    ssize = 64
    wsize = 64
    gthresh = 4
    gripper_segments = segment(gripper_time, gripper_pos, base_thresh=thresh, segment_size=ssize, window_size=wsize, grace_thresh=gthresh, plot=False)
    
    segments = probabilistically_combine([joint_segments, traj_segments, frc_segments, gripper_segments], len(traj_pos), wsize, n_samples=20, n_pass=3, plot=True)
    
    plt.rcParams['figure.figsize'] = (9, 7)
    fig = plt.figure()
    fig.suptitle('Trajectory')
    ax = plt.axes(projection='3d')
    for i in range(len(segments)-1):
        ax.plot3D(traj_pos[segments[i]:segments[i+1], 0], traj_pos[segments[i]:segments[i+1], 1], traj_pos[segments[i]:segments[i+1], 2], label="Segment " + str(i+1), lw=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.legend()
    plt.tight_layout()
    plt.show()

#Example process using a 2D trajectory with a single data stream
def main2d():
    np.random.seed(6)
    [[sm_t, sm_x, sm_y], [unsm_t, unsm_x, unsm_y], [norm_t, norm_x, norm_y]] = scr2.read_demo_h5('r_shape.h5', 3)
    norm_y = -norm_y
    demo = np.hstack((np.reshape(norm_x, (len(norm_x), 1)), np.reshape(norm_y, (len(norm_y), 1))))
    
    thresh = 0.16
    ssize = 64
    wsize = 16
    segments = segment(norm_t, demo, base_thresh=thresh, segment_size=ssize, window_size=wsize, plot=True)
    
    segments = probabilistically_combine([segments], len(demo), wsize, n_samples=10, n_pass=2, plot=True)
    print('Final Segments')
    print(segments)
    
    fig = plt.figure(figsize=(6, 6))
    for i in range(len(segments)-1):
        plt.plot(demo[segments[i]:segments[i+1], 0], demo[segments[i]:segments[i+1], 1], lw=5, c=colors[i+1], label="Segment " + str(i+1))
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='lower center')
    plt.show()
    
if __name__ == '__main__':
    print("Example of 2D trajectory")
    main2d()
    print("Example of 3D trajectory")
    main3d()