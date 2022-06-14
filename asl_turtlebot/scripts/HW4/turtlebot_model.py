import numpy as np

EPSILON_OMEGA = 1e-3

def compute_dynamics(xvec, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute g, Gx, Gu
    # HINT: To compute the new state g, you will need to integrate the dynamics of x, y, theta
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.


    ########## Code ends here ##########
    x, y, theta = xvec
    V, om = u

    theta_n = theta + om * dt

    cos_n = np.cos(theta_n)
    cos = np.cos(theta)
    sin_n = np.sin(theta_n)
    sin = np.sin(theta)

    if np.abs(om) < EPSILON_OMEGA:
        g = xvec + dt * V * np.array([cos, sin, 0])
    else:
        g = xvec + np.array([ V/om * (sin_n - sin),
                             -V/om * (cos_n - cos),
                              om * dt])
    if not compute_jacobians:
        return g

    if np.abs(om) > EPSILON_OMEGA:
        Gx = np.array([ [1, 0, V/om*(cos_n - cos)],
                        [0, 1, V/om*(sin_n - sin)],
                        [0, 0, 1]])
        Gu = np.array([ [1/om * (sin_n - sin),
                         -V/(om*om)*(sin_n - sin) + V/om*(cos_n)*dt],
                        [-1/om*(cos_n - cos),
                         V/(om*om)*(cos_n - cos) + V/om*(sin_n)*dt],
                        [0, dt]])
    else:
        Gx = np.array([ [1, 0, -V*(sin + sin_n)/2*dt],
                        [0, 1, V*(cos + cos_n)/2*dt],
                        [0, 0, 1]])
        Gu = np.array([ [cos, V*(sin + sin_n)/2*(dt/2)],
                        [sin, -V*(cos + cos_n)/2*(dt/2)],
                        [0, 1]]) * dt

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx
    # HINT: Calculate the pose of the camera in the world frame (x_cam, y_cam, th_cam), a rotation matrix may be useful.
    # HINT: To compute line parameters in the camera frame h = (alpha_in_cam, r_in_cam), 
    #       draw a diagram with a line parameterized by (alpha,r) in the world frame and 
    #       a camera frame with origin at x_cam, y_cam rotated by th_cam wrt to the world frame
    # HINT: What is the projection of the camera location (x_cam, y_cam) on the line r? 
    # HINT: To find Hx, write h in terms of the pose of the base in world frame (x_base, y_base, th_base)


    ########## Code ends here ##########
    def Rot(th):
        return np.array([ [np.cos(th), -np.sin(th)],
                            [np.sin(th), np.cos(th)] ])

    x, y, theta = x
    x_base, y_base, theta_base = tf_base_to_camera

    x_diff, y_diff = Rot(theta) @ np.array([x_base, y_base])

    x_cam, y_cam = x + x_diff, y + y_diff
    theta_cam = theta + theta_base

    alpha_in_cam = alpha - (theta + theta_base)
    r_in_cam = r - (Rot(-alpha) @ np.array([x_cam, y_cam]))[0]
    # = r - (x + x_diff) np.cos(-alpha) - (y + y_diff) np.sin(-alpha)

    h = np.array([alpha_in_cam, r_in_cam])
    if not compute_jacobian:
        return h

    Hx = np.array([ [0, 0, -1],
                    [-np.cos(alpha), -np.sin(alpha),
                    (y_base*np.cos(alpha)-x_base*np.sin(alpha)) * np.cos(theta) +
                    (x_base*np.cos(alpha)+y_base*np.sin(alpha)) * np.sin(theta)]])
    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
