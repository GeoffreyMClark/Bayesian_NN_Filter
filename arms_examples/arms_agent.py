import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper, gym_wrapper
from robosuite.controllers import load_controller_config
import math



test_num = '00'
data_dir = '/home/geoffrey/Research/data/arms/test_'+test_num+'/'

# load default controller parameters for Operational Space Control (OSC)
controller_config = load_controller_config(default_controller="OSC_POSE")
controller_config['control_delta']=True
controller_config['damping_ratio']=4
controller_config['uncouple_pos_ori']=True



# create an environment to visualize on-screen
# env = GymWrapper(
env = suite.make(
    "TwoArmHandover",
    robots=["UR5e", "UR5e"],             # load a Sawyer robot and a Panda robot
    gripper_types="default",                # use default grippers per robot arm
    controller_configs=controller_config,   # each arm is controlled using OSC
    env_configuration="single-arm-opposed", # (two-arm envs only) arms face each other
    has_renderer=True,                      # on-screen rendering
    render_camera="frontview",              # visualize the "frontview" camera
    has_offscreen_renderer=False,           # no off-screen rendering
    control_freq=100,                        # 20 hz control for applied actions
    horizon=1000,                            # each episode terminates after 200 steps
    use_object_obs=False,                   # no observations needed
    use_camera_obs=False,                   # no observations needed
)

observation = env.reset()
initial_pos = observation['robot0_eef_pos']
initial_quat = observation['robot0_eef_quat']



def euler_from_quaternion(quat):
        x=quat[0]; y=quat[1]; z=quat[2]; w=quat[3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        return np.asarray([-pitch_y, roll_x, yaw_z]) # in radians

def circleShift(angle):
    temp = np.where(angle>(np.pi), angle-(2*np.pi), angle)
    adjusted = np.where(temp<(-np.pi), temp+(2*np.pi), temp)
    return adjusted

 
def controller(t, observation):
    h_pos = env._hammer_pos
    h_ang = euler_from_quaternion(env._hammer_quat)
    init_ang = euler_from_quaternion(initial_quat)
    eef_ang = euler_from_quaternion(observation['robot0_eef_quat'])
    h_move_ang = np.asarray([init_ang[0], init_ang[1,], h_ang[2]-np.pi/2])
    end_pose = initial_pos + np.asarray([0.0,0.5,0.3])
    end_ang = init_ang + np.asarray([np.pi/2,0.0,0.0])

    # square = np.sin(t/150)

    if t <= 0:
        r0_goal_pos = initial_pos - observation['robot0_eef_pos']
        r0_goal_angle = circleShift((init_ang-eef_ang))
        r0_goal_gripper = np.asarray([-1])
        r1_goal_pos = np.asarray([0.0,0.0,0.0])
        r1_goal_angle = np.asarray([0.0,0.0,0.0])
        r1_goal_gripper = np.asarray([0.0])

    elif t > 0 and t <= 250:
        r0_goal_pos = h_pos - observation['robot0_eef_pos']
        r0_goal_angle = circleShift((h_move_ang-eef_ang))
        r0_goal_gripper = np.asarray([-1])
        r1_goal_pos = np.asarray([0.0,0.0,0.0])
        r1_goal_angle = np.asarray([0.0,0.0,0.0])
        r1_goal_gripper = np.asarray([0.0])

    elif t > 250 and t <= 300:
        r0_goal_pos = h_pos - observation['robot0_eef_pos']
        r0_goal_angle = circleShift((h_move_ang-eef_ang))
        r0_goal_gripper = np.asarray([1])
        r1_goal_pos = np.asarray([0.0,0.0,0.0])
        r1_goal_angle = np.asarray([0.0,0.0,0.0])
        r1_goal_gripper = np.asarray([0.0])

    if t >300 and t<= 370:
        r0_goal_pos = (h_pos+np.asarray([0.0,0.0,0.3])) - observation['robot0_eef_pos']
        r0_goal_angle = circleShift((h_move_ang-eef_ang))
        r0_goal_gripper = np.asarray([1])
        r1_goal_pos = np.asarray([0.0,0.0,0.0])
        r1_goal_angle = np.asarray([0.0,0.0,0.0])
        r1_goal_gripper = np.asarray([0.0])

    if t >370:
        r0_goal_pos = end_pose - observation['robot0_eef_pos']
        r0_goal_angle = circleShift((end_ang-eef_ang)*0.5)
        r0_goal_gripper = np.asarray([1])
        r1_goal_pos = np.asarray([0.0,0.0,0.0])
        r1_goal_angle = np.asarray([0.0,0.0,0.0])
        r1_goal_gripper = np.asarray([0.0])

    print(init_ang)
    print(eef_ang)
    print(r0_goal_angle)
    print(eef_ang+(r0_goal_angle))
    print('')
    action = np.concatenate((r0_goal_pos*20, r0_goal_angle, r0_goal_gripper, r1_goal_pos*15, r1_goal_angle, r1_goal_gripper), axis=0)
    return action



if __name__ == "__main__":

    for t in range(5000):
        env.render()
        
        action = controller(t, observation)
        
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

# to do:
# 1. add control interpolator
# 2. add controller policy_freq

