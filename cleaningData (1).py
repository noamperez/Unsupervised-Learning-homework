import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

data = pd.read_csv("AuditData.csv", encoding='unicode_escape', low_memory=False)
none_kinds = ['?', 'NULL', 'None', 'NA', '', '#N/A']
data = data.replace(to_replace=none_kinds, value=np.nan)

"""
features_to_remove = (data.isnull().sum() / data.shape[0]) > 0.2
features_to_remove = features_to_remove[features_to_remove].index.values
print(features_to_remove)
=values = {'Auditor_name': 'Unknown', 'Auditor_id': 0, 'Big4': -1, 'Audit_opinion': 'Unknown'}
data = data.fillna(value=values)
data = data.dropna()
data.to_csv('AuditData.csv')
"""


