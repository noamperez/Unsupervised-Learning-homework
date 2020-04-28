import numpy as np
import pandas as pd
import xlsxwriter
import jinja2
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
# print(data.shape)
# print(data.dtypes)
# none_kinds = ['?', 'NULL', 'None', 'NA', '', '#N/A']
# data = data.replace(to_replace=none_kinds, value=np.nan)
# #
# #
# features_to_remove = (data.isnull().sum() / data.shape[0]) > 0.2
# features_to_remove = features_to_remove[features_to_remove].index.values
# print(features_to_remove)
# values = {'Auditor_name': 'Unknown', 'Auditor_id': 0, 'Big4': -1, 'Audit_opinion': 'Unknown'}
# data = data.fillna(value=values)
# data = data.dropna()
# data.to_csv('AuditData.csv')
#
data.drop('Unique_identifier', axis=1, inplace=True)
data.drop('Year_of_establishment', axis=1, inplace=True)
data.drop('Industry_code', axis=1, inplace=True)
data.drop('Company_name', axis=1, inplace=True)
data.drop('CEO_id', axis=1, inplace=True)
data.drop('Auditor_id', axis=1, inplace=True)
data.drop('Company_id', axis=1, inplace=True)


my_colors = ['b', 'r', 'g', 'y', 'k','m']
#
#
# Audit_opinion = data.groupby('Audit_opinion').size().plot(kind="bar", color=my_colors, fontsize=15)
# vals = Audit_opinion.get_yticks()
# Audit_opinion.set_yticklabels(['{:,.0%}'.format(x / data.shape[0]) for x in vals])
# Audit_opinion
# plt.show()
#
#
# Restructuring = data.groupby('Restructuring').size().plot(kind="bar", color=my_colors , title= 'Restructuring')
# vals = Restructuring.get_yticks()
# Restructuring.set_yticklabels(['{:,.0%}'.format(x / data.shape[0]) for x in vals])
# Restructuring
# plt.show()
#
# Bankruptcy = data.groupby('Bankruptcy').size().plot(kind="bar", color=my_colors , title= 'Bankruptcy')
# vals = Bankruptcy.get_yticks()
# Bankruptcy.set_yticklabels(['{:,.0%}'.format(x / data.shape[0]) for x in vals])
# Bankruptcy
# plt.show()
#
#
# Legal_form = data.groupby('Legal_form').size().plot(kind="bar", color=my_colors , fontsize=15)
# print(Legal_form)
# vals = Legal_form.get_yticks()
# Legal_form.set_yticklabels(['{:,.0%}'.format(x / data.shape[0]) for x in vals])
# Legal_form
# plt.show()
#
#
# Year = data.groupby('Year').size().plot(kind="bar",  color= my_colors , title= 'Year')
# vals = Year.get_yticks()
# Year.set_yticklabels(['{:,.0%}'.format(x / data.shape[0]) for x in vals])
# Year
# plt.show()

#
# Big4 = data.groupby('Big4').size().plot(kind="bar",  color= my_colors , title= 'Big4')
# vals = Big4.get_yticks()
# Big4.set_yticklabels(['{:,.0%}'.format(x / data.shape[0]) for x in vals])
# Big4
# plt.show()


# # Auditor_name = data.groupby('Auditor_name').size().plot(kind="bar",  color= my_colors , title= 'Auditor_name')
# # vals = Auditor_name.get_yticks()
# # Auditor_name.set_yticklabels(['{:,.0%}'.format(x / data.shape[0]) for x in vals])
# # Auditor_name
# # plt.show()
# # CEO_id = data.groupby('CEO_id').size().plot(kind="bar",  color= my_colors , title= 'CEO_id')
# # vals = CEO_id.get_yticks()
# # CEO_id.set_yticklabels(['{:,.0%}'.format(x / data.shape[0]) for x in vals])
# # CEO_id
# # plt.show()
#
#
# pd.crosstab(data['Year'],data['Bankruptcy']).plot(kind='bar')
# plt.show()
#
# pd.crosstab(data['Year'],data['Restructuring']).plot(kind='bar')
# plt.show()
#
# pd.crosstab(data['Big4'],data['Restructuring']).plot(kind='bar')
# plt.show()
#
# pd.crosstab(data['Big4'],data['Legal_form']).plot(kind='bar')
# plt.show()
#
# pd.crosstab(data['Big4'],data['Audit_opinion']).plot(kind='bar')
# plt.show()
#
# pd.crosstab(data['Year'],data['Audit_opinion']).plot(kind='bar')
# plt.show()
#

data.drop('Auditor_name', axis=1, inplace=True)


print(data.shape)
data = data[data.Bankruptcy!=1]
print(data.shape)
data.drop('Bankruptcy', axis=1, inplace=True)
print(data.shape)


col=data.columns
categorical_data = data.select_dtypes('object')
print(categorical_data.dtypes)
numerical_data=[]
for columns in col:
    if columns not in categorical_data:
        numerical_data.append(columns)

writer= pd.ExcelWriter('corr matrix.xlsx', engine = 'xlsxwriter')
corr=data[numerical_data].corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2).to_excel(writer)
writer.save()
df = pd.read_csv("corr matrix.csv", encoding='unicode_escape', low_memory=False, index_col=[0])
corr_list1=[]
corr_list2=[]
corr_list3=[]
for i in df.columns:
    idx = df.index[df[i]>=0.8]
    if len(idx)>10:
        corr_list1.append(idx)
    idx_min= df.index[df[i]>=0.1]
    if len(idx_min)<=2:
        corr_list2.append(idx_min)
    idx_equal= df.index[df[i] ==1]
    if len(idx_equal)>=2:
        corr_list3.append(idx_equal)
print('the feature that have more then 10 other features with corolation above 0.8 is:\n' + str(corr_list1) + 'and the number of the features is:\n' + str(len(corr_list1)))
print('the feature that have corrolation equals 1 with other features is:\n' + str(corr_list3) + 'and the number of the features is:\n' + str(len(corr_list3)))
print('the feature that have not more then 1 other features with corolation above 0.1 is:\n' +  str(corr_list2) + 'and the number of the features is:\n' + str(len(corr_list2)))

print(corr_list2)
print(len(corr_list2))
print(corr_list3)
print(len(corr_list3))


# # arr = data["Legal_form"].unique()

data['Audit_opinion'] = data['Audit_opinion'].replace('Adverse', 0)
data['Audit_opinion'] = data['Audit_opinion'].replace('Disclaimer', 1)
data['Audit_opinion'] = data['Audit_opinion'].replace('Unqualified', 2)
data['Audit_opinion'] = data['Audit_opinion'].replace('Qualified', 3)
data['Audit_opinion'] = data['Audit_opinion'].replace('Unknown', -1)

data['Legal_form'] = data['Legal_form'].replace('Corporation', 0)
data['Legal_form'] = data['Legal_form'].replace('Limited liability', 1)
data['Legal_form'] = data['Legal_form'].replace('Public utility', 2)
data['Legal_form'] = data['Legal_form'].replace('Cooperative', 3)
data['Legal_form'] = data['Legal_form'].replace('Socially owned enterprise', 4)
data['Legal_form'] = data['Legal_form'].replace('Partnership', 5)
data['Legal_form'] = data['Legal_form'].replace('Branch of a foreign company', 6)
data['Legal_form'] = data['Legal_form'].replace('Limited partnership', 7)
data['Legal_form'] = data['Legal_form'].replace('Other', 8)

data['Restructuring'] = data['Restructuring'].replace('No',0)
data['Restructuring'] = data['Restructuring'].replace('Yes',1)

data['Big4'] = data['Big4'].replace('No',0)
data['Big4'] = data['Big4'].replace('Yes',1)
data['Big4'] = data['Big4'].replace('Unknown',-1)

data.to_csv('Data.csv')
