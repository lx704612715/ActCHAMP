# this script is about trajectory segmentation of a latch lock using CHAMPT algorithm

import os
import rospkg
import yaml
from contact_lfd.LfDusingEC.scripts.data_loader import DemoDataLoader
from pytransform3d.trajectories import pqs_from_transforms
from pytransform3d.batch_rotations import batch_quaternion_xyzw_from_wxyz
import rospy
from changepoint_detection.srv import DetectChangepoints, DetectChangepointsRequest
from changepoint_detection.msg import DataPoint


def makeDetectRequest(req):
    try:
        dc = rospy.ServiceProxy('changepoint_detection/detect_changepoint_detections', DetectChangepoints)
        resp = dc(req)
        return resp
    except rospy.ServiceException as e:
        print("Service call failed:", e)

pkg_path = rospkg.RosPack().get_path('contact_lfd')
config_path = os.path.join(pkg_path, 'src/contact_lfd/LfDusingEC/config/locks_experiment/augmentation_chain_lock.yaml')
config = yaml.safe_load(open(config_path, "r"))
config["experiment_path"] = "/home/lx/experiments/lx/exp_LfDIP/"
config["experiment_name"] = "DrawerIROS23Video"
data_loader = DemoDataLoader(config, load_data=True, filtering=True, load_pkl=True, exp_name_from_ros=False)

filtered_seg_demonstration = data_loader.get_motion_force_patterns_for_each_demo(cut_off_rate=10,
                                                                                 min_dist_threshold=0.0005,
                                                                                 min_ori_threshold=0.005)
filtered_demonstration = data_loader.merge_seg_motion_force_patterns_for_each_demo(filtered_seg_demonstration)

filtered_data = filtered_seg_demonstration[1][2]
obj_ht_ee = filtered_data["base_ht_ee"].reshape(-1, 4, 4)

obj_pqs_ee = pqs_from_transforms(obj_ht_ee)   # x, y, z, w, rx, ry, rz
# champt requires the data format starting with rx
obj_pqs_ee[:, 3:] = batch_quaternion_xyzw_from_wxyz(obj_pqs_ee[:, 3:])

req = DetectChangepointsRequest()
req.data = [DataPoint(x) for x in obj_pqs_ee]
req.model_type = 'changepoint_detection/ArticulationFitter'
req.cp_params.len_mean = 50.0
req.cp_params.len_sigma = 10.  # 5.0
req.cp_params.min_seg_len = 20  # 3
req.cp_params.max_particles = 10
req.cp_params.resamp_particles = 10
resp = makeDetectRequest(req)

for seg in resp.segments:
    print("Model:", seg.model_name)
    print("Length:", seg.last_point - seg.first_point + 1)
    print("Start:", seg.first_point)
    print("End:", seg.last_point)
    for i in range(len(seg.model_params)):
        print("  ", seg.param_names[i])
        print(":", seg.model_params[i])

print("seg")


