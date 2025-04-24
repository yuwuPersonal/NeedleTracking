# Import tip locations per frame, process and visualize the data
#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import sys

center_per_frame = np.load("center_per_frame.npy")
video = cv.VideoCapture("needle_sample.mp4")

# Calculate microns per pixel using the 2.5-mm bodycut
if not video.isOpened():
    print("Can not open the video!")
    sys.exit()
else:
    rec, frame = video.read()
    bodycut = cv.selectROI(frame, showCrosshair=True)
    cv.waitKey(0)
    cv.destroyAllWindows()
    length_per_px = 2.5e3 / bodycut[-1] # microns

# %%
center_um = center_per_frame * length_per_px
center_um = center_um - center_um[0]

fps = video.get(cv.CAP_PROP_FPS)
t_range = np.arange(center_um.shape[0]) / fps

# fig = plt.figure(0)
# for which_tip in range(center_um.shape[1]):
#     rect, ax = fig.add_axes([0, 0, 30, 5000])
#     ax.plot(t_range, center_um[:, which_tip, -1])

plt.plot(t_range, center_um[:, :, -1])
plt.xlim((0, 40))
plt.xlabel("Time(s)"), plt.ylabel("Travel distance(um)")
# %%
