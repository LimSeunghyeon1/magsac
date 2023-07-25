import numpy as np
import matplotlib.pyplot as plt
import cv2
import pymagsac
import time
from copy import deepcopy 
import open3d as o3d

pcd = o3d.io.read_point_cloud('/home/tidy/plane_detection/OCID-dataset/ARID20/table/bottom/seq07/pcd/result_2018-08-21-14-10-43.pcd')
pcd = pcd.remove_non_finite_points()
def verify_pymagsac(points, use_magsac_plus_plus, h, w, sampler_id):
    plane, mask = pymagsac.findPlane3D(
        np.ascontiguousarray(points), 
        w, h,
        probabilities = [],
        sampler = sampler_id,
        use_magsac_plus_plus = use_magsac_plus_plus,
        sigma_th = 0.07)
    print("line", plane, "plane", mask)
    print (deepcopy(mask).astype(np.float32).sum(), 'inliers found')
    return plane, mask

p = np.asarray(pcd.points)
col = np.zeros_like(p)
t = time.time()
plane, mask = verify_pymagsac(p, True, 480, 640, 1)

col[mask] = np.array([1,0,0])
a = o3d.geometry.PointCloud()
a.points = o3d.utility.Vector3dVector(p)
a.colors = o3d.utility.Vector3dVector(col)
o3d.visualization.draw_geometries([a])
print("da", time.time() - t)