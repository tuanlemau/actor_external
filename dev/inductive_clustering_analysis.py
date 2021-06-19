#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Retrieve median rating 604 x 13
# compute plot methods


# In[1]:


# How do I increase the cell width of the Jupyter/ipython notebook in my browser? https://stackoverflow.com/a/34058270
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
from datetime import date
dateStamp = date.today().strftime("%y%m%d")


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


# needed imports# neede 
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import fcluster
import numpy as np

# some setting for this notebook to actually show the graphs inline
# you probably won't need this
get_ipython().run_line_magic('matplotlib', 'inline')
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float


# In[36]:


import sklearn as sk


# In[37]:


print(sk.__version__)


# In[38]:


import scipy 


# In[39]:


print(scipy.__version__)


# In[4]:


EMO_CAT=['Amusement', 'Anger', 'Awe', 'Contempt', 'Disgust', 'Embarrassment', 'Fear', 'Happiness', 'Interest', 'Pride', 'Sadness', 'Shame', 'Surprise']


# In[5]:


def get_table_cophenet_corrcoeff_for_all_methods_and_distance_metrics(matrix):
    X=matrix  
    
    # METHODS:
        # note difference between average & centroid
        # average: average pairwise
        # centroid: dist btw two cluster centroid
    methodLIST=['single','complete','average','weighted','centroid','median','ward']
    distanceMetricLIST=['euclidean','cityblock','seuclidean',
                        'sqeuclidean','cosine','correlation','hamming',
                        'jaccard','chebyshev','canberra','braycurtis',
                        'mahalanobis','yule','matching','dice','kulsinski',
                        'rogerstanimoto','russellrao','sokalmichener',
                        'sokalsneath','wminkowski'
                       ]
    c_mat=np.zeros((len(methodLIST),len(distanceMetricLIST)))

    for i,iMETHOD in enumerate(methodLIST):
        for j,iMETRIC in enumerate(distanceMetricLIST):
            try:
                Z_temp = linkage(X,method=iMETHOD,metric=iMETRIC)
                c_mat[i,j], coph_dists = cophenet(Z_temp, pdist(X))
            except:
                c_mat[i,j]=-1
    return pd.DataFrame(c_mat,columns=distanceMetricLIST,index=methodLIST)


# In[6]:


EMO_ORDER_FOR_ANALYSIS=['Awe', 'Fear', 'Anger', 'Pride', 'Shame', 'Disgust', 'Sadness',
       'Contempt', 'Interest', 'Surprise', 'Amusement', 'Happiness',
       'Embarrassment']


# In[7]:


EMO_LIST_SHORT=['AMU', 'ANG', 'AWE', 'CON', 'DIS', 'EMB', 'FEA', 'HAP', 'INT', 'PRI', 'SAD', 'SHA', 'SUR']


# In[8]:


# 1. COMPARE ACROSS ALL METHODS & DISTANACE METRIC


# In[9]:


AU_LIST=[str(i) for i in range(1,28)]


# In[10]:


def get_reference_clustered_AU_data_from_previous_code():
    dfAU_ref=pd.read_csv('../data/for_import/dfAU.csv')
    dfAU_ref.set_index('index',inplace=True)
    return dfAU_ref.set_index('survey_questions_id')

dfAU_ref = get_reference_clustered_AU_data_from_previous_code()
CORRECT_ORDER= dfAU_ref.index


# In[11]:


def retrieve_AUs_raw_value(dfSource):
    AU_LIST_and_ID=AU_LIST + ['id']

    df_AUs=dfSource[AU_LIST_and_ID].set_index('id')
    df_AUs.index.name='survey_questions_id'
    return df_AUs.sort_index()

def transformAUpatternToAUlist(AUpattern): return [int(i) for i in AUpattern[AUpattern==1].index]
def get_individualRatings_from(dataframe,condition='/survey-so'): return dataframe[dataframe['survey_condition']==condition]

df_individualRatings_all = pd.read_csv('../data/core_data/survey_data.csv')
df_so=get_individualRatings_from(df_individualRatings_all)
df_eachScenario=df_so.groupby('survey_questions_id')
dfMedian=df_eachScenario.median()[EMO_CAT]
dfMedian=dfMedian[EMO_ORDER_FOR_ANALYSIS].loc[CORRECT_ORDER]

df_scenes=pd.read_csv('../data/core_data/survey_questions.csv')
df_AUs=retrieve_AUs_raw_value(df_scenes)
df_AUs=df_AUs.loc[CORRECT_ORDER]


# In[12]:


result_allMethods=get_table_cophenet_corrcoeff_for_all_methods_and_distance_metrics(dfMedian.values)
# result_allMethods.to_csv('../data/auto_generated/'+dateStamp+ 'hierarchical_clustering_method_metric.csv')


# In[13]:


# 2. PERFORM ANALYSIS WITH SELECTED METHOD & METRIC on the matrix of 604 scenarios x 13 median rating on emotion categories 
METHOD_SELECTED = 'average'
METRIC_SELECTED = 'euclidean'

X = dfMedian.loc[CORRECT_ORDER].values
Z = linkage(X, method=METHOD_SELECTED, metric=METRIC_SELECTED)
c, coph_dists = cophenet(Z, pdist(X))


# In[14]:


# retrieve Action Unit
# get assigned cluster with given numClust


# In[15]:


NUM_CLUST = 80
def assign_each_scenario_to_a_cluster_by_its_pose_ActionUnits(Z,NUM_CLUST): return fcluster(Z,NUM_CLUST,criterion='maxclust')
df_AUs['cluster']=assign_each_scenario_to_a_cluster_by_its_pose_ActionUnits(Z,NUM_CLUST)
dfMedian['cluster']=assign_each_scenario_to_a_cluster_by_its_pose_ActionUnits(Z,NUM_CLUST)


# In[16]:


def computeMatch(AUvec1,AUvec2,indexVec=False):       
    if(indexVec):
        row=refVec=np.zeros(27)
#         refVec=np.zeros(27)
        row[AUvec1-1]=1
        refVec[AUvec2-1]=1
    else:
        row=AUvec1
        refVec=AUvec2
        
    sumOverlap=np.sum(np.logical_and(row,refVec))
    numerator=sumOverlap*2

    denominator=np.sum(row>0)+np.sum(refVec>0)
    #print('sumOverlap:Num:dec',sumOverlap,numerator,denominator)
    return numerator/denominator


# In[17]:


def get_all_row_belonging_to_cluster(i,dfSource): return dfSource[dfSource.cluster==i]
def get_number_of_poses_in_cluster(cluster): n,numPose= cluster.shape;return numPose
def get_pairwise_match_score_array(dfCluster):
    clusSize = get_number_of_poses_in_cluster(dfCluster)
    arr=list()
    for i in range(0,clusSize):
        for j in range(i+1,clusSize):
            arr.append(computeMatch(dfCluster.iloc[:,i].values,dfCluster.iloc[:,j].values))
    return arr

def get_pairwise_match_score_array_2(dfCluster):
    import actionUnitMatching as matcher
    clusSize = get_number_of_poses_in_cluster(dfCluster)
    arr=list()
    for i in range(0,clusSize):
        for j in range(i+1,clusSize):
            arr.append(matcher.compute_match_score_method1(dfCluster.iloc[i]['AUpattern'],dfCluster.iloc[j]['AUpattern']))
    return arr

def compute_intraClusterMedian(pairwiseMatchscoreArr): return np.median(np.nan_to_num(np.array(pairwiseMatchscoreArr)))


# In[18]:


def compute_intraClusterMedian_Matchscore_for(iCluster,dfSource):
    dfEachClust = get_all_row_belonging_to_cluster(iCluster,dfSource)
    dfEachClust = dfEachClust[AU_LIST].T
    clusSize = get_number_of_poses_in_cluster(dfEachClust)
    if(clusSize>1):
        arr=get_pairwise_match_score_array(dfEachClust)
        return compute_intraClusterMedian(arr)
    else:
        return -1


# In[19]:


def extract_all_scenes_with_median_rating_belonging_to(iCluster,dfSource=dfMedian):
    df=get_all_row_belonging_to_cluster(iCluster,dfSource)
    df=df.drop('cluster',axis=1)
    df.columns=[emo[:3].upper() for emo in df.columns]
    df=df[df.columns.sort_values()]   
    return df


# In[20]:


NUM_CLUST=80


# In[21]:


intraClustMed=np.zeros(NUM_CLUST)
Xc=dict() #store individual cluster's rating data
pattern=dict()
for iCluster in range(1,NUM_CLUST+1):
    intraClustMed[iCluster-1]=compute_intraClusterMedian_Matchscore_for(iCluster,dfSource=df_AUs)
    pattern[iCluster]=get_all_row_belonging_to_cluster(iCluster,dfSource=df_AUs).drop('cluster',axis=1)
    Xc[iCluster]=extract_all_scenes_with_median_rating_belonging_to(iCluster,dfSource=dfMedian)


# In[22]:


def getSalientCluster(dfCluster,salientTHRESHOLD=1):
    return [i for i in dfCluster.columns[dfCluster.median()>=salientTHRESHOLD]]


# In[23]:


def format_salient_cluster(salientCluster,strItems=''):
    for index,i in enumerate(salientCluster): strItems+='+'*(index>0) + str(i)         
    return strItems


# In[24]:


def extract_and_format_salient_cluster(dfCluster,salientTHRESHOLD,strItems=''):
    return format_salient_cluster(getSalientCluster(dfCluster,salientTHRESHOLD))


# In[25]:


def retrieve_AU_cluster_string(dfCluster,strQ='QID='):        
    for ind,item in enumerate(dfCluster.index): strQ+=','*(ind>0)+ str(item)
    return strQ


# In[26]:


def get_each_row_as_panda_series(AUIndex,EmoIndex,intraClustMed,strQ,iCluster,Xc):
    return pd.Series({'AU Clusters':AUIndex,
               'Emo clusters':EmoIndex,
               'Intra-clust SCore': '{:.2f}'.format(intraClustMed[iCluster-1]),
                'Scenarios' : strQ,
               'Cluster':iCluster,
               'Cluster size':Xc[iCluster].shape[0],
            })


# In[27]:


def get_AUvtSeries(patternCluster,salientTHRESHOLD):
    AUind=[int(i)-1 for i in getSalientCluster(patternCluster,salientTHRESHOLD)]
    AUvec_temp=np.zeros(27)
    AUvec_temp[AUind]+=1 
    AUvtSeries=pd.Series(AUvec_temp,index=AU_LIST)
    AUvtSeries.index=[int(i) for i in AUvtSeries.index]
    return AUvtSeries


# In[28]:


INTENT_THRES=2;EMO_THRES=1
pdFormatTable=pd.DataFrame() 

for iCluster in range(1,NUM_CLUST+1):
    if(len(Xc[iCluster])>1): 
        EmoIndex=extract_and_format_salient_cluster(Xc[iCluster],salientTHRESHOLD=EMO_THRES)
        AUIndex=extract_and_format_salient_cluster(pattern[iCluster][AU_LIST],salientTHRESHOLD=INTENT_THRES)
    elif(len(Xc[iCluster])==1):
        EmoIndex='-';AUIndex='-'            
    strQ=retrieve_AU_cluster_string(Xc[iCluster])
    
   
    AUvtSeries = get_AUvtSeries(pattern[iCluster][AU_LIST],salientTHRESHOLD=INTENT_THRES)
    pdSTemp=get_each_row_as_panda_series(AUIndex,EmoIndex,intraClustMed,strQ,iCluster,Xc)
    
    pdFormatTable=pd.concat([pdFormatTable,pd.concat([pdSTemp,AUvtSeries])],axis=1)


# In[29]:


pdFormatTable.T.set_index('Cluster')[['Cluster size','Intra-clust SCore','AU Clusters','Emo clusters']].sort_values('Intra-clust SCore',ascending=False)# .to_csv('../data/auto_generated/TableS3'+dateStamp+'.csv')


# In[30]:


def leaveLabelFunc(i): return leave_labels[i]


# In[31]:


leaveData=pdFormatTable.T.set_index('Cluster')


# In[32]:


R=dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=NUM_CLUST,  # show only the last NUM_CLUST merged clusters
#     leaf_label_func=leaveLabelFunc, #add customized labelling callback function (In [176]) for truncated leaf (cool!)
    leaf_rotation=0,
    leaf_font_size=16.,
    color_threshold=2,above_threshold_color='grey',
    orientation='left', #just add orientation to make dendogram horizontal
)


# In[33]:


# *** Most important assignment for leaf label ***
#change name from temp to leave_labels for readability
# * in front indicates that the cluster intra-cluster match score is equal or greater than 0.4
leave_labels = {R["leaves"][ii]:'>>>' * (float(leaveData['Intra-clust SCore'][ii+1])>=0.4)        + str(ii+1) +'.'        + '[AU ' + leaveData['AU Clusters'][ii+1] +'] ['        + leaveData['Emo clusters'][ii+1] +'] '#         + '(N='+str(clusterSize[ii+1])+')' \
                for ii in range(len(R["leaves"]))}\
        


# In[34]:


plt.figure(figsize=(15,35))
plt.title('Hierarchical Clustering Dendrogram\n',fontsize=40)
plt.ylabel('Clusters',fontsize=20)
plt.xlabel('Cluster Distance',fontsize=20)
R=dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=NUM_CLUST,  # show only the last NUM_CLUST merged clusters
    leaf_label_func=leaveLabelFunc, #add customized labelling callback function (In [176]) for truncated leaf (cool!)
    leaf_rotation=0,
    leaf_font_size=16.,
    color_threshold=2,above_threshold_color='grey',
    orientation='left', #just add orientation to make dendogram horizontal
)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




