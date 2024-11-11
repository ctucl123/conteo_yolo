import numpy as np
dets =[
        [np.float32(996.69324), np.float32(754.2272), np.float32(1383.4403), np.float32(1027.9598), np.float32(0.93939495)],
        [np.float32(1242.4861), np.float32(605.61475), np.float32(1473.7205), np.float32(863.12634), np.float32(0.86046064)]
        ]
output_results = np.array(dets, dtype=np.float32)


print(output_results.shape[1])
scores = output_results[:, 4]
print(scores)