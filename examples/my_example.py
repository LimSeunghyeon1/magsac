import numpy as np
import matplotlib.pyplot as plt
import cv2
import pymagsac
import pygcransac
import time
from copy import deepcopy 
import open3d as o3d
from sklearn.cluster import DBSCAN

pcd = o3d.io.read_point_cloud('/home/tidy/plane_detection/OCID-dataset/ARID20/table/bottom/seq07/pcd/result_2018-08-21-14-10-43.pcd')
pcd = pcd.remove_non_finite_points()
pcd = pcd.voxel_down_sample(0.01)
def verify_pymagsac(points, use_magsac_plus_plus, h, w, sampler_id):
    plane, mask = pymagsac.findPlane3D(
        np.ascontiguousarray(points), 
        w, h,
        probabilities = [],
        sampler = sampler_id,
        use_magsac_plus_plus = use_magsac_plus_plus,
        sigma_th = 0.07)
    print (deepcopy(mask).astype(np.float32).sum(), 'inliers found')
    return plane, mask

def verify_pygcransac(points, h, w, sampler_id):
    points = np.ascontiguousarray(points)
    
    plane, mask = pygcransac.findPlane3D(
        np.ascontiguousarray(points), 
        w, h,
        probabilities = [],
        threshold = .005,
        conf = .95,
        max_iters = 1000,
        min_iters = 100,
        spatial_coherence_weight=0.1,
        use_sprt=True,
        sampler = sampler_id)
    print (deepcopy(mask).astype(np.float32).sum(), 'inliers found')
    return plane, mask


point_list = []
p = np.asarray(pcd.points)

t = time.time()
while True:
    # plane, mask = verify_pymagsac(p, True, 480, 640, 1)
    plane, mask = verify_pygcransac(p, 480, 640, 1)
    masked_xyz = p[mask]
    if len(masked_xyz) == 0:
        break
    db = DBSCAN(min_samples=1).fit(masked_xyz)
    db_lbls = db.labels_
    for i in range(db_lbls.max()+1):
        masked_masked_xyz = masked_xyz[db_lbls == i]
        col = np.zeros_like(masked_masked_xyz)
        col[:] = np.random.random(3)
        a = o3d.geometry.PointCloud()
        a.points = o3d.utility.Vector3dVector(masked_masked_xyz)
        a.colors = o3d.utility.Vector3dVector(col)
        point_list.append(a)
    p = p[~mask]
    print("p", len(p))
    if len(p) == 0:
        break
col = np.zeros_like(p)
a = o3d.geometry.PointCloud()
a.points = o3d.utility.Vector3dVector(p)
a.colors = o3d.utility.Vector3dVector(col)
point_list.append(a)

print("da", time.time() - t)
o3d.visualization.draw_geometries(point_list)