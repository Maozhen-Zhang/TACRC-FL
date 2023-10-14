import numpy as np
from sklearn.preprocessing import StandardScaler

data = np.array([
    [3.0, 0.3],
    [4.0, 0.2],
    [5.0, 0.5]
])

zscore_scaler = StandardScaler()
data_zscore_dis = zscore_scaler.fit_transform(data)
