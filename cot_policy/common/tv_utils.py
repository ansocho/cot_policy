import numpy as np
import matplotlib.pyplot as plt
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.metrics import dtw


def dba_mean_trajectory(trajectories):
    """
    Calculate the DBA mean trajectory from a list of trajectories.
    Each trajectory is assumed to be of shape (time_steps, features).
    """
    # Convert trajectories to the appropriate shape for the dtw_barycenter_averaging function
    # Ensure each trajectory has the same shape for the barycenter averaging.
    trajectories = [t.reshape(t.shape[0], -1) for t in trajectories]

    # Compute the DTW barycenter averaging (DBA) of the trajectories
    mean_trajectory = dtw_barycenter_averaging(
        trajectories, max_iter=50
    )  # Adjust max_iter as needed

    return mean_trajectory


def calculate_dtw_distance(trajectory_1, trajectory_2):
    """
    Calculate the DTW distance between two trajectories.
    Each trajectory is assumed to be a numpy array of shape (time_steps, features).
    """
    distance = dtw(trajectory_1, trajectory_2)
    return distance


def calculate_dtw_variance_from_dba(trajectories):
    """
    Calculate the DTW variance of the trajectories with respect to the DBA mean trajectory.
    """
    # Step 1: Calculate the DBA mean trajectory
    mean_trajectory = dba_mean_trajectory(trajectories)

    # Step 2: Calculate the DTW distance from each trajectory to the mean trajectory
    dtw_distances = []
    for trajectory in trajectories:
        dtw_dist = calculate_dtw_distance(
            trajectory.reshape(trajectory.shape[0], -1), mean_trajectory
        )
        dtw_distances.append(dtw_dist)

    # Step 3: Calculate the variance of the DTW distances
    dtw_distances = np.array(dtw_distances)
    dtw_variance = np.mean(dtw_distances**2)

    return dtw_variance


if __name__ == "__main__":
    env_name = "maze2d-custom-3-v0"
    # env_name = "coffee_d1"
    policy_type = "conditional_ot_clust_state_policy"
    # policy_type = "conditional_ot_clust_policy"
    policy_type = "fm_policy"
    num_inference_steps = 3

    filename = (
        policy_type + "_" + str(num_inference_steps) + "_" + env_name + "_actions.npy"
    )
    rewards_filename = (
        policy_type + "_" + str(num_inference_steps) + "_" + env_name + "_rewards.npy"
    )
    all_rewards_filename = (
        policy_type
        + "_"
        + str(num_inference_steps)
        + "_"
        + env_name
        + "_all_rewards.npy"
    )

    trajectories = np.load("misc/" + filename)
    rewards = np.load("misc/" + rewards_filename).squeeze()
    all_rewards = np.load("misc/" + all_rewards_filename).squeeze()

    trajectory_list = []
    print(rewards.shape)
    for i in range(trajectories.shape[0]):  # For each env trajectory
        if rewards[i] > 0:
            rollout_completed_idx = np.where(all_rewards[i] == 1)[0][0]
            trajectory_list.append(trajectories[i])  # [:rollout_completed_idx])

    dtw_variance = calculate_dtw_variance_from_dba(trajectory_list)
    print(f"DTW Variance: {dtw_variance}")
    print(f"Success rate: {len(trajectory_list)/rewards.shape[0]}")
