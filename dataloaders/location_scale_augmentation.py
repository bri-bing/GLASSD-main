import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

class LocationScaleAugmentation(object):
    def __init__(self, vrange=(0.,1.), background_threshold=0.01, nPoints=4, nTimes=100000):
        self.nPoints=nPoints
        self.nTimes=nTimes
        self.vrange=vrange
        self.background_threshold=background_threshold

    def get_monotonic_b_spline(self, coordinates, k=2, nTimes=100):
        x = np.array([p[0] for p in coordinates])
        y = np.array([p[1] for p in coordinates])

        t = np.linspace(0,1,len(x) - k + 1)
        t = np.concatenate(([0] * k, t, [1] * k))

        spline_x = BSpline(t,x,k)
        spline_y = BSpline(t,y,k)

        t_vals = np.linspace(0,1,nTimes)
        x_vals = spline_x(t_vals)
        y_vals = spline_y(t_vals)

        return x_vals,y_vals

    def non_linear_transformation(self, inputs, k, nPoints, inverse=False, inverse_prop=0.5):
        start_point, end_point = inputs.min(), inputs.max()
        coordinate = [[start_point,start_point],[end_point,end_point]]
        # x_min,y_min = coordinate[0]
        # x_max,y_max = coordinate[1]
        for _ in range(len(coordinate), nPoints):
            x_new = random.random()
            y_new = random.random()
        
            coordinate.insert(-1, [x_new, y_new])
    
        tmp = sorted([p[0] for p in coordinate],reverse=False)
        for i in range(len(coordinate)):
            coordinate[i][0] = tmp[i]
        if inverse and random.random() <= inverse_prop:
            tmp = sorted([p[1] for p in coordinate],reverse=True)
            for i in range(len(coordinate)):
                coordinate[i][1] = tmp[i]
        else:
            tmp = sorted([p[1] for p in coordinate],reverse=False)
            for i in range(len(coordinate)):
                coordinate[i][1] = tmp[i]
        #print(end_point)

        x_val,y_val = self.get_monotonic_b_spline(coordinate,k)

        return np.interp(inputs, x_val, y_val)

    def location_scale_transformation(self, inputs, slide_limit=20):
        scale = np.array(max(min(random.gauss(1, 0.1), 1.1), 0.9), dtype=np.float32)
        location = np.array(random.gauss(0, 0.5), dtype=np.float32)
        location = np.clip(location, self.vrange[0] - np.percentile(inputs, slide_limit), self.vrange[1] - np.percentile(inputs, 100 - slide_limit))
        return np.clip(inputs*scale + location, self.vrange[0], self.vrange[1])

    def Global_Location_Scale_Augmentation(self, image):
        image=self.non_linear_transformation(image, 3, 4, inverse=False)
        image=self.location_scale_transformation(image).astype(np.float32)
        return image

    def Local_Location_Scale_Augmentation(self,image, mask):
        output_image = np.zeros_like(image)

        mask = mask.astype(np.int32)

        output_image[mask == 0] = self.location_scale_transformation(self.non_linear_transformation(image[mask==0], 3, 4, inverse=True, inverse_prop=1))

        for c in range(1,np.max(mask)+1):
            if (mask==c).sum()==0:continue
            output_image[mask == c] = self.location_scale_transformation(self.non_linear_transformation(image[mask == c], 3, 4, inverse=True, inverse_prop=0.3))

        if self.background_threshold>=self.vrange[0]:
            output_image[image <= self.background_threshold] = image[image <= self.background_threshold]

        return output_image
