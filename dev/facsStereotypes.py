#!/usr/bin/env python
# coding: utf-8

# In[204]:


import pandas as pd


# In[205]:


EMO_LIST_SHORT=['index', 'AMU', 'ANG', 'AWE', 'CON', 'DIS', 'EMB', 'FEA', 'HAP', 'INT', 'PRI', 'SAD', 'SHA', 'SUR']


# In[238]:


class AUdataReadWrite():
    def __init__(
        self,
        PATH_ROOT='../data/for_import/ref_AU_auto/',
        EMO_CAT_LABEL='EmotionCategory',  
        INDEX_LABEL='Index',
        AU_COL_LABEL='AUPattern'
    ):
        self.PATH_ROOT=PATH_ROOT
        self.l_EMOCAT=EMO_CAT_LABEL
        self.l_INDEX=INDEX_LABEL
        self.l_AUPattern=AU_COL_LABEL
        
    def get_file_list(self):
        import os
        return [f[:-4] for f in os.listdir(self.PATH_ROOT)]
        
    def load_AUdata_into_currentFormat(self,fileName):
        def prep_panda_table_to_convert_to_dict(df): return df.set_index([self.l_EMOCAT,self.l_INDEX]).T.to_dict()
        def get_list_of_emoCats(data):
            return [emo for emo in data[self.l_EMOCAT].unique()]
        def convert_to_dict_in_desiredFormat(prep,emoCats):
            def initDict():  
                tempDict=dict()              
                for emoCat in emoCats: tempDict[emoCat]=dict() 
                return tempDict
            def turn_into_int_list(strList): return [int(i) for i in strList.strip('][').split(', ')] 

            tempDict=initDict()
            for (emoCat,index),p in prep.items(): tempDict[emoCat][index]=turn_into_int_list(p[self.l_AUPattern])
            return tempDict

        dat=pd.read_csv(self.PATH_ROOT+fileName+'.csv')
        prep= prep_panda_table_to_convert_to_dict(dat)
        emoCats=get_list_of_emoCats(dat)
        return convert_to_dict_in_desiredFormat(prep,emoCats)
    
    def write_to_csv(self,AU_VEC,fileName):
        d=pd.DataFrame(pd.DataFrame(AU_VEC).T.stack())

        d.index.names=[self.l_EMOCAT,self.l_INDEX]
        d.columns=[self.l_AUPattern]

        d.to_csv(self.PATH_ROOT+fileName+'.csv')


# In[239]:


rw=AUdataReadWrite()


# In[240]:


rw.get_file_list()


# In[241]:


REF_AU_VEC=rw.load_AUdata_into_currentFormat('refAU_13_ekman')
REF_AU_VEC_plusICPonly = rw.load_AUdata_into_currentFormat('refAU_13_ekman_plusICPonly')
REF_AU_USA_VEC=rw.load_AUdata_into_currentFormat('refAU_13_Cordaro_onlyUSA')
REF_AU_ICP_USA=rw.load_AUdata_into_currentFormat('refAU_13_Cordaro_ICPandUSA')
REF_AU_VEC_SPECIFICITY=rw.load_AUdata_into_currentFormat('refAU_6_cat_specificity_analysis')


# In[207]:


PATH_CORDARO_AUS ='../data/for_import/cordaro2017_au_icp_accent_usa.csv'
def retrieve_CordaroAUdata_from_file(filePath):
    import pandas as pd
    return pd.read_csv(filePath)
retrieve_CordaroAUdata_from_file(PATH_CORDARO_AUS).head()


# In[208]:


def remremoveAUgreaterThan27fromDict(REF_VEC):
    return {emocat: {index: removeAUgreaterThan27(AUpattern) for index, AUpattern in obj.items()} 
        for emocat, obj in REF_VEC.items()}

def removeAUgreaterThan27(AUpattern):
    return [AU for AU in AUpattern if AU <= 27]

REF_AU_USA_VEC_removedAUabove27 = remremoveAUgreaterThan27fromDict(REF_AU_USA_VEC)


# In[209]:


assert(removeAUgreaterThan27([7, 12, 53, 6, 25, 64]) == [7, 12, 6, 25])
assert(removeAUgreaterThan27([4, 17, 54, 24, 64])== [4,17,24])

