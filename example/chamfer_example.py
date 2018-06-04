import numpy as np
from template_ffd.eval.normalize import normalized
from template_ffd.metrics.np_impl import np_metrics

gt_cloud=np.load("gt_cloud.npy")
inf_cloud=np.load("inf_cloud.npy")
scale=np.load("scale.npy")
offset=np.load("offset.npy")
gt_cloud=np.squeeze(gt_cloud)
inf_cloud=np.squeeze(inf_cloud)
values=[]
for i in xrange(0,803):
    single_gt_cloud=gt_cloud[i:i+1,:,:]
    single_inf_cloud=inf_cloud[i:i+1,:,:]
    single_scale=scale[i:i+1]
    single_offset=offset[i:i+1,:]
    rescaled_gt_cloud = normalized(single_gt_cloud,single_offset,single_scale)
    rescaled_inf_cloud = normalized(single_inf_cloud,single_offset,single_scale)
    res=np_metrics.chamfer(rescaled_gt_cloud, rescaled_inf_cloud) / 1024
    print i,res
    values.append(res)
print np.mean(values)
