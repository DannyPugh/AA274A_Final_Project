import numpy as np
import scipy.interpolate

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    #raise NotImplementedError # REPLACE THIS FUNCTION WITH YOUR IMPLEMENTATION

    #path passed from astar as a tuple by reconstruct_path, move to array
    path = np.array(path)

   #determine nominal time from desired velocity given Velocity is change in distance/change in time
    #this much be done for each point because they are not equal distances apart

    time = np.zeros(len(path))
        
    for i in range(1, len(path)):
        #change in distance
        difference = path[i] - path[i-1]
        dist = np.linalg.norm(difference)
        #nominal in time by definition of average velocity
        delt_t = dist/V_des
        #time at position given initial condition at previous pos
        time[i] = delt_t + time[i-1]

        #path_index = np.linspace(0, len(path)-1, len(path))

    #use splerlp to find overlay a cubic coefficient, smoothing to account for repeated x values
    #tck = scipy.interpolate.splrep(path[:,1], path[:,0], s = alpha)
    tckx = scipy.interpolate.splrep(time, path[:,0], s = alpha)
    tcky = scipy.interpolate.splrep(time, path[:,1], s = alpha)


    #use splev to computer all of the desired derivatives, 2 derivatives therefore der = 2
    #keeping N segments as given in the description and defined as the row size of path

    #this is to maintain N as the number of spaces for t_smoothed
    t_smoothed = np.arange(0, time[-1], dt)

    #computer x_d, y_d and each derivative uding scipy and iterating der (order of derivative to computer)
    y_d = scipy.interpolate.splev(t_smoothed, tcky)
    x_d = scipy.interpolate.splev(t_smoothed, tckx)
    xd_d = scipy.interpolate.splev(t_smoothed, tckx, der = 1)
    yd_d = scipy.interpolate.splev(t_smoothed, tcky, der = 1)
    xdd_d = scipy.interpolate.splev(t_smoothed, tckx, der = 2)
    ydd_d = scipy.interpolate.splev(t_smoothed, tcky, der = 2)

    #computer theta using the kinematic equations from problem set 1
    theta_d = np.arctan2(yd_d, xd_d)

    traj_smoothed = np.stack([x_d, y_d, theta_d, xd_d, yd_d, xdd_d, ydd_d]).transpose()
    ########## Code ends here ##########

    return traj_smoothed, t_smoothed
