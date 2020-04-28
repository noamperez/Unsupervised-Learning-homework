import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph
from sklearn import preprocessing
from sklearn import decomposition
from scipy.cluster.hierarchy import dendrogram
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from random import randint

data = pd.read_csv("Data.csv", encoding='unicode_escape', low_memory=False)

# one hot encoder
data['Audit_opinion'] = data['Audit_opinion'].replace(0,'Adverse')
data['Audit_opinion'] = data['Audit_opinion'].replace(1,'Disclaimer')
data['Audit_opinion'] = data['Audit_opinion'].replace(2,'Unqualified')
data['Audit_opinion'] = data['Audit_opinion'].replace(3,'Qualified')
data['Audit_opinion'] = data['Audit_opinion'].replace(-1,'Unknown')

data['Legal_form'] = data['Legal_form'].replace(0,'Corporation')
data['Legal_form'] = data['Legal_form'].replace(1,'Limited liability')
data['Legal_form'] = data['Legal_form'].replace(2,'Public utility')
data['Legal_form'] = data['Legal_form'].replace(3,'Cooperative')
data['Legal_form'] = data['Legal_form'].replace(4,'Socially owned enterprise')
data['Legal_form'] = data['Legal_form'].replace(5,'Partnership')
data['Legal_form'] = data['Legal_form'].replace(6,'Branch of a foreign company')
data['Legal_form'] = data['Legal_form'].replace(7,'Limited partnership')
data['Legal_form'] = data['Legal_form'].replace(8,'Other')
Audit_opinion_dummies = pd.get_dummies(data['Audit_opinion'])
Legal_form_dummies = pd.get_dummies(data['Legal_form'])
data = data.drop('Audit_opinion', axis=1)
data = data.drop('Legal_form', axis=1)
data = pd.concat([data, Audit_opinion_dummies,Legal_form_dummies ], axis=1)

#data slicing for unknown and known 'Audit opinion'
data_check=data[data.Unknown==1]
data=data[data.Unknown!=1]
data.drop('Unknown', axis=1, inplace=True)

my_colors = ['b', 'r', 'g', 'y', 'k', 'm']

###################################################
normalize_data = preprocessing.normalize(data)

##PCA-3D,2D and checking ratio variance
################################
# # check when we get to 90% that represent the data
# k=2
# stop=False
# while stop==False:
#     pca = PCA(n_components=k)
#     pca.fit(normalize_data)
#     pca_data = pca.transform(normalize_data)
#     x=sum(pca.explained_variance_ratio_)
#     k+=1
#     if x>0.9:
#         stop=True
#         print(pca.explained_variance_ratio_)
#         print('eighen value that represent 90% of the data = ' +str(len(pca.explained_variance_ratio_))+ '\n'
#         + 'and the ratio is ' +str(x))
#
# def var_plot(score,coeff,labels=None):
#     xs = score[:,0]
#     ys = score[:,1]
#     n = coeff.shape[0]
#     scalex = 1.0/(xs.max() - xs.min())
#     scaley = 1.0/(ys.max() - ys.min())
#     plt.scatter(xs * scalex, ys * scaley, cmap= 'rgb')
#     for i in range(n):
#         plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
#         if labels is None:
#             plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
#         else:
#             plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
# plt.xlim(-1,1)
# plt.ylim(-1,1)
# plt.xlabel("PC{}".format(1))
# plt.ylabel("PC{}".format(2))
# plt.grid()
#
#
# #2D
# pca = PCA(n_components=2)
# pca.fit(normalize_data)
# pca_data=pca.transform(normalize_data)
# ax=plt.scatter(pca_data[:, 0], pca_data[:, 1],  cmap=plt.cm.nipy_spectral,edgecolor='k')
# plt.title("PCA-2D")
# plt.axis('tight')
# plt.show()
#
#
# #plot the imprtance feature in PC
# var_plot(pca_data ,np.transpose(pca.components_[0:2, :]))
# plt.show()

# #3D
# pca = PCA(n_components=3)
# pca.fit(normalize_data)
# pca_data=pca.transform(normalize_data)
# fig = plt.figure(1, figsize=(4, 3))
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
# ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], cmap=plt.cm.nipy_spectral,edgecolor='k')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.title("PCA-3D")
# plt.axis('tight')
# plt.show()










# # TSNE dimentional reduce
# normalize_data =preprocessing.normalize(data)
# T_sne_model = manifold.TSNE(n_components=3,init='pca',random_state=0, perplexity= 50, n_iter=500)
# T_sne_model_results = T_sne_model.fit_transform(normalize_data[1:])
# print(type(T_sne_model_results))
# mid = int(len(T_sne_model_results)/2)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(T_sne_model_results[:,0],T_sne_model_results[:,1],T_sne_model_results[:,2],cmap=plt.cm.nipy_spectral,edgecolor='k')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.title("t-SNE")
# plt.axis('tight')
# plt.show()




#####################################################
# # #Kmeans cluster algorithm
#
# def compute_bic_kmeans(kmeans,X):
#     """
#     Computes the BIC metric for a given clusters
#     Parameters:
#     -----------------------------------------
#     kmeans:  List of clustering object from scikit learn
#     X     :  multidimension np array of data points
#     Returns:
#     -----------------------------------------
#     BIC value
#     """
#     centers = [kmeans.cluster_centers_]
#     labels  = kmeans.labels_
#     #number of clusters
#     m = kmeans.n_clusters
#     # size of the clusters
#     n = np.bincount(labels)
#     #size of data set
#     N, d = X.shape
#     #compute variance for all clusters beforehand
#     cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X.iloc[np.where(labels == i)], [centers[0][i]],
#              'euclidean')**2) for i in range(m)])
#     const_term = 0.5 * m * np.log(N) * (d+1)
#     BIC = np.sum([n[i] * np.log(n[i]) -n[i] * np.log(N) -
#              ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
#              ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term
#
#     return(BIC)
#
# #################################################################
# normalize_data =pd.DataFrame(preprocessing.normalize(data))
# elbow_scores = {}
# silhouette_scores = {}
# for k in range(2, 9):
#     kmeans_model = KMeans(n_clusters=k, n_jobs=-1)
#     kmeans_model.fit(normalize_data.values)
#     elbow_scores.update({k: kmeans_model.inertia_})
#     silhouette_score_val = silhouette_score(normalize_data, kmeans_model.labels_)
#     silhouette_scores.update({k: silhouette_score_val})
#
#
# plt.plot(list(elbow_scores.keys()), list(elbow_scores.values()))
# plt.xlabel("Number of clusters")
# plt.ylabel("Elbow score")
# plt.show()
#
# plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()))
# plt.xlabel("Number of clusters")
# plt.ylabel("Silhouette score")
# plt.show()
#
# # perform PCA for the best clustring
# value=list(silhouette_scores.values())
# k= value.index(max(value))+2
#
# kmeans_model = KMeans(n_clusters=k, n_jobs=-1)
# kmeans_model.fit(normalize_data)
# kmeans_pca = PCA(n_components=k)
# kmeans_pca_data = kmeans_pca.fit_transform(normalize_data)
# analysis_data = data.copy()
# analysis_data["clusters"] = kmeans_model.labels_
#
# for i in range(k):
#     print('Kmeans - cluster ' + str(i) + ' = ' + str(sum(analysis_data["clusters"]==i)))
#
# # plot 3D scatter
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter([i[0] for i in kmeans_pca_data], [i[1] for i in kmeans_pca_data], [i[2] for i in kmeans_pca_data],
#            c=[my_colors[j] for j in kmeans_model.labels_])
# plt.show()
# fig2 = plt.figure()
# ax2 = Axes3D(fig2)
# ax2.scatter([i[1] for i in kmeans_pca_data], [i[0] for i in kmeans_pca_data], [i[2] for i in kmeans_pca_data],c=[my_colors[j] for j in kmeans_model.labels_])
# plt.show()
#
#
#
# #one hot encode reverse for graphs
#
# analysis_data['Audit_opinion'] = analysis_data[['Adverse','Disclaimer', 'Unqualified', 'Qualified']].idxmax(axis=1)
# analysis_data['Legal_form'] = analysis_data[['Corporation','Limited liability','Public utility','Cooperative','Socially owned enterprise','Partnership','Branch of a foreign company','Limited partnership','Other']].idxmax(axis=1)
#
# #graph of features vs clusters
# pd.crosstab(analysis_data['clusters'], analysis_data['Audit_opinion']).plot(kind='bar')
# plt.show()
#
# pd.crosstab(analysis_data['clusters'], analysis_data['Restructuring']).plot(kind='bar')
# plt.show()
#
# pd.crosstab(analysis_data['clusters'], analysis_data['Big4']).plot(kind='bar')
# plt.show()
#
# pd.crosstab(analysis_data['clusters'], analysis_data['Year']).plot(kind='bar')
# plt.show()
#
# pd.crosstab(analysis_data['clusters'], analysis_data['Legal_form']).plot(kind='bar')
# plt.show()
#
# #BIC compute
# print('The BIC score of k_means with ' str(k) + ' centroids is: ' + str(compute_bic_kmeans(kmeans_model.fit(normalize_data),normalize_data)))


#Agglomerative_model
###############################################################

# normalize_data =pd.DataFrame(preprocessing.normalize(data))
# silhouette_scores = {}
# for k in range(2, 9):
#     knn = kneighbors_graph(normalize_data, 10, n_jobs=-1)
#     Agglomerative_model = AgglomerativeClustering(n_clusters=k, connectivity=knn)
#     Agglomerative_model.fit(normalize_data.values)
#     silhouette_score_val = silhouette_score(normalize_data, Agglomerative_model.labels_)
#     silhouette_scores.update({k: silhouette_score_val})
#
#
# #plotting silouette score graph
# plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()))
# plt.xlabel("Number of clusters")
# plt.ylabel("Silhouette score")
# plt.show()
#
# #perform PCA for the best clustering
# value=list(silhouette_scores.values())
# k= value.index(max(value))+2
#
# knn = kneighbors_graph(normalize_data, 10, n_jobs=-1)
# Agglomerative_model = AgglomerativeClustering(n_clusters=k, connectivity=knn)
# Agglomerative_model.fit(normalize_data.values)
# agglomerative_pca = PCA(n_components=3)
# agglomerative_pca_data = agglomerative_pca.fit_transform(normalize_data)
#
# analysis_data = data.copy()
# analysis_data["clusters"] = Agglomerative_model.labels_
#
# for i in range(k):
#     print('Agglomerative - cluster ' + str(i) + ' = ' + str(sum(analysis_data["clusters"]==i)))
#
# # plot 3D scatter
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter([i[0] for i in agglomerative_pca_data], [i[1] for i in agglomerative_pca_data], [i[2] for i in agglomerative_pca_data],
#            c=[my_colors[j] for j in Agglomerative_model.labels_])
# plt.show()
# fig2 = plt.figure()
# ax2 = Axes3D(fig2)
# ax2.scatter([i[1] for i in agglomerative_pca_data], [i[0] for i in agglomerative_pca_data], [i[2] for i in agglomerative_pca_data],c=[my_colors[j] for j in Agglomerative_model.labels_])
# plt.show()
#
# #one hot encode reverse for graphs
#
# analysis_data['Audit_opinion'] = analysis_data[['Adverse','Disclaimer', 'Unqualified', 'Qualified']].idxmax(axis=1)
# analysis_data['Legal_form'] = analysis_data[['Corporation','Limited liability','Public utility','Cooperative','Socially owned enterprise','Partnership','Branch of a foreign company','Limited partnership','Other']].idxmax(axis=1)
#
# #graph of features vs clusters
# pd.crosstab(analysis_data['clusters'], analysis_data['Audit_opinion']).plot(kind='bar')
# plt.show()
#
# pd.crosstab(analysis_data['clusters'], analysis_data['Restructuring']).plot(kind='bar')
# plt.show()
#
# pd.crosstab(analysis_data['clusters'], analysis_data['Big4']).plot(kind='bar')
# plt.show()
#
# pd.crosstab(analysis_data['clusters'], analysis_data['Year']).plot(kind='bar')
# plt.show()
#
# pd.crosstab(analysis_data['clusters'], analysis_data['Legal_form']).plot(kind='bar')
# plt.show()
#
# #dendogram plot
#
# def plot_dendrogram(model, **kwargs):
#     # Create linkage matrix and then plot the dendrogram
#
#     # create the counts of samples under each node
#     counts = np.zeros(model.children_.shape[0])
#     n_samples = len(model.labels_)
#     for i, merge in enumerate(model.children_):
#         current_count = 0
#         for child_idx in merge:
#             if child_idx < n_samples:
#                 current_count += 1  # leaf node
#             else:
#                 current_count += counts[child_idx - n_samples]
#         counts[i] = current_count
#
#     linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
#
#     # Plot the corresponding dendrogram
#     dendrogram(linkage_matrix, **kwargs)
#
# normalize_data =pd.DataFrame(preprocessing.normalize(data))
# Agglomerative_model_dendogram = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
# plot_dendrogram(Agglomerative_model_dendogram.fit(normalize_data.values), truncate_mode='level', p=3)
# plt.xlabel("Number of points in node.")
# plt.show()

#gaussian_mixture_model
######################################################
#
# normalize_data =pd.DataFrame(preprocessing.normalize(data))
# silhouette_scores = {}
# for k in range(2,9):
#     gaussian_mixture_model = GaussianMixture(n_components=k)
#     gaussian_mixture_model.fit(normalize_data.values)
#     gaussian_mixture_model_labels = gaussian_mixture_model.fit_predict(normalize_data)
#     silhouette_score_val = silhouette_score(normalize_data,gaussian_mixture_model_labels)
#     silhouette_scores.update({k: silhouette_score_val})
# #
# # plotting silouette score graph
# plt.plot(list(silhouette_scores.keys()),list(silhouette_scores.values()))
# plt.xlabel("Number of clusters")
# plt.ylabel("Silhouette score")
# plt.show()
#
# # perform PCA for the best clustering
# value=list(silhouette_scores.values())
# k= value.index(max(value))+2
# gaussian_mixture_model = GaussianMixture(n_components=k)
# gaussian_mixture_model_labels = gaussian_mixture_model.fit_predict(normalize_data)
# # perform PCA for plotting
# gaussian_mixture_pca = PCA(n_components=3)
# gaussian_mixture_pca_data = gaussian_mixture_pca.fit_transform(normalize_data)
#
# analysis_data = data.copy()
# analysis_data["clusters"] = gaussian_mixture_model_labels
# for i in range(k):
#     print('GMM - cluster ' + str(i) + ' = ' + str(sum(analysis_data["clusters"]==i)))
#
# # plot 3D scatter
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter([i[0] for i in gaussian_mixture_pca_data], [i[1] for i in gaussian_mixture_pca_data], [i[2] for i in gaussian_mixture_pca_data],
#            c=[my_colors[j] for j in gaussian_mixture_model_labels])
# plt.show()
#
# fig2 = plt.figure()
# ax2 = Axes3D(fig2)
# ax2.scatter([i[1] for i in gaussian_mixture_pca_data], [i[0] for i in gaussian_mixture_pca_data], [i[2] for i in gaussian_mixture_pca_data],
#             c=[my_colors[j] for j in gaussian_mixture_model_labels])
# plt.show()
#
# #one hot encode reverse for graphs
#
# analysis_data['Audit_opinion'] = analysis_data[['Adverse','Disclaimer', 'Unqualified', 'Qualified']].idxmax(axis=1)
# analysis_data['Legal_form'] = analysis_data[['Corporation','Limited liability','Public utility','Cooperative','Socially owned enterprise','Partnership','Branch of a foreign company','Limited partnership','Other']].idxmax(axis=1)
#
# #graph of features vs clusters
# pd.crosstab(analysis_data['clusters'], analysis_data['Audit_opinion']).plot(kind='bar')
# plt.show()
#
# pd.crosstab(analysis_data['clusters'], analysis_data['Restructuring']).plot(kind='bar')
# plt.show()
#
# pd.crosstab(analysis_data['clusters'], analysis_data['Big4']).plot(kind='bar')
# plt.show()
#
# pd.crosstab(analysis_data['clusters'], analysis_data['Year']).plot(kind='bar')
# plt.show()
#
# pd.crosstab(analysis_data['clusters'], analysis_data['Legal_form']).plot(kind='bar')
# plt.show()
#
# #BIC compute
# print('The BIC score of GMM with ' + str(k) +  ' centroids is: ' + str(gaussian_mixture_model.bic(normalize_data)))
