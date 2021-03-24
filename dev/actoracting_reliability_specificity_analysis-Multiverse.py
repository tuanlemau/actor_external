#!/usr/bin/env python
# coding: utf-8

# In[1]:


# How do I increase the cell width of the Jupyter/ipython notebook in my browser? https://stackoverflow.com/a/34058270
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os

from datetime import date,datetime
dateStamp = date.today().strftime("%y%m%d")

np.random.seed(0)


# ### Steps: goal: characterize the match score between poses & stereotypic pattern 
# 
# **data input** 
#     
#     read_individualRating_from_condition(scenarioAlone)
#     
# **common across analyses**
#     
#     represent_pose_with_ActionUnit_pattern
#     compute_median_ratings
#     
#     compute_mean_estimates_and_CI
# 
# **reliability analysis**
# 
#     assign_stereotypicEmotionCategory_to_each_scenario ~ retrieveHighestRatedEmotionCategoryForEachScenario
#         define_assignment_criteria
#             emotionCategory_with_highest_median_rating 
#                 break_tie_basedOn_narrower_interquartileRange 
#                     implementation:
#                         ~ intermediate step: compute JointProxyWhereHighestRatedCategoryIsMinimumValue = dfMedianProxy+dfIQR
#                         ~ extractCategoryWithHighestMedianSmallestIQR
# 
#     compare_between_each_pose_and_stereotypicPattern_using_matchScore 
#         define_comparison_metric
#         compute_matchScore_using_defined_metric
#         
#     
#     +characterize_matchScore_using_statisticalMethods        
# 
# **specificity_analysis** 
# 
#     represent_pose_with_ActionUnit_pattern ~ dfAU_binaryActivation=obtainActionUnitActivationDataframeBasedOn(df_original,ACTIVATION_THRESHOLD)  
#     
#     assign_poses_into_emotion_category_by_AUstereotypes
#         pick_matching_poses_forEach_AUstereotypes_variants 
#             compute_match_score_for_all_poses_against_a_variant
#             pick_poses_with_score_above_specificityThreshold
#         combine_matching_poses_across_variants_of_AU_stereotypes
#         retain_unique_set_of_matching_poses_by_removing_overlaps
#         
#     determine_if_pose_specific_to_assignged_emotion_category
#         is_pose_NOT_specific_to_assigned_emotion_category
#             is_rating_of_assigned_emotion_category_BELOW_presenceThreshold
#             is_rating_of_another_emotion_category_ATLEAST_presenseThreshold    
#         
#     compute_false_positive_rate_for_each_emotion_category
#         count_total_poses_assigned_to_emotion_category
#         count_false_positive_poses_per_assigned_to_emotion_category
#         false_positive = count_false_positive_poses_per_assigned_to_emotion_category/ count_total_poses_assigned_to_emotion_category
#       

# In[3]:


### MAIN ###


# In[4]:


class Matcher():
    def __init__(self,AU_PATTERN_FILE='refAU_13_ekman_plusICPonly'):
            ### MAIN ###
        self.compute_match = {
            'method1' : self.compute_match_score_method1,
            'method2' : self.compute_match_score_method2,
            'method1_sim_addAU' : self.compute_match_score_method1_simulate_additionalAU_baseline_median,
            'method2_sim_addAU' : self.compute_match_score_method2_simulate_additionalAU_baseline_median,
            'method1_allAddAU': self.compute_match_score_method1_additionalAU_alwaysOn,
            'method2_allAddAU': self.compute_match_score_method2_additionalAU_alwaysOn,
            'method1_sim_addAU_baseline_max' : self.compute_match_score_method1_simulate_additionalAU_baseline_max,
            'method2_sim_addAU_baseline_max' : self.compute_match_score_method2_simulate_additionalAU_baseline_max,
        }

        from facsStereotypes import AUdataReadWrite
        rw=AUdataReadWrite()
        self.STEREOTYPES = rw.load_AUdata_into_currentFormat(AU_PATTERN_FILE)
        
        self.compute_matchScore_across_stereotypeVariants={
            'median': self.compute_median_matchScore_across_stereotypeVariants,
            'max' : self.compute_max_matchScore_across_stereotypeVariants,
            'median-cordaro': self.compute_median_matchScore_across_stereotypeVariants_addCordaroUSA,
        }

    @staticmethod
    def compute_match_score_method1(target,ref,v=False): 
        ''' #method 1 - coderReliabilityMethod: jointly activated AU x 2 / all activated AU across target & reference pattern '''
        print('method1') if(v) else ''
        try:
            return len(set(target).intersection(set(ref)))*2/(len(target)+len(ref))   
        except:
            return 1 # fully matched - neutral

    @staticmethod
    def compute_match_score_method2(target,ref,v=False): 
        ''' #method 2 - directMatchMethod: overlapped activated AU / activated AU of reference pattern '''
        print('method2') if(v) else ''
        return len(set(target).intersection(set(ref)))/len(ref)


    ### CONSIDERING ADDITION AU ###
    @staticmethod
    def get_number_of_AUs_above_27(ref): AU27vec=[i for i in ref if i <=27];return len(ref)-len(AU27vec)
    
    def compute_match_score_method1_simulate_additionalAU_baseline_median(self,target,ref,v=False,AU_BASELINE=.104):
        return self.compute_match_score_method1_simulate_additionalAU(target,ref,v=False,AU_BASELINE=AU_BASELINE)
    
    def compute_match_score_method2_simulate_additionalAU_baseline_median(self,target,ref,v=False,AU_BASELINE=.104):
        return self.compute_match_score_method2_simulate_additionalAU(target,ref,AU_BASELINE=AU_BASELINE,v=False)
    
    def compute_match_score_method1_simulate_additionalAU_baseline_max(self,target,ref,v=False,AU_BASELINE=.651):
        return self.compute_match_score_method1_simulate_additionalAU(target,ref,v=False,AU_BASELINE=AU_BASELINE)
    
    def compute_match_score_method2_simulate_additionalAU_baseline_max(self,target,ref,v=False,AU_BASELINE=.651):
        return self.compute_match_score_method2_simulate_additionalAU(target,ref,v=False,AU_BASELINE=AU_BASELINE)

    def compute_match_score_method1_simulate_additionalAU(self,target,ref,AU_BASELINE,v=False):
        ''' #method 1, simulate activated AU above 27 (1000 simulations), randomly assigned whether AU above 27 within stereotype pattern is activated in target pattern'''
        n27=self.get_number_of_AUs_above_27(ref)

        def randomize_number_of_AUoverlap_above_AU27(totalAUabove27,AU_BASELINE): 
            return np.sum(np.random.rand(1000,totalAUabove27)<AU_BASELINE,axis=1)
        o27Arr= randomize_number_of_AUoverlap_above_AU27(n27,AU_BASELINE)

        def computeMatch(n): return (len(set(target).intersection(set(ref)))*2+n*2)/(len(target)+len(ref)+n+n27)    
        ms= [computeMatch(n) for n in o27Arr]

        return np.median(ms)

    def compute_match_score_method2_simulate_additionalAU(self,target,ref,AU_BASELINE,v=False):
        ''' #method 2, simulate activated AU above 27 (1000 simulations), randomly assigned whether AU above 27 within stereotype pattern is activated in target pattern '''
        n27=self.get_number_of_AUs_above_27(ref)

        def randomize_number_of_AUoverlap_above_AU27(totalAUabove27,AU_BASELINE = AU_BASELINE): 
            return np.sum(np.random.rand(1000,totalAUabove27)<.65,axis=1)
        o27Arr= randomize_number_of_AUoverlap_above_AU27(n27,AU_BASELINE) 

        def computeMatch(n): return (len(set(target).intersection(set(ref)))+n)/(len(ref)+n27)   
        ms= [computeMatch(n) for n in o27Arr]

        return np.median(ms) 

    def compute_match_score_method1_additionalAU_alwaysOn(self,target,ref,v=False):
        n27=self.get_number_of_AUs_above_27(ref)
        return (len(set(target).intersection(set(ref)))*2+n27*2)/(len(target)+len(ref)+n27*2)  

    def compute_match_score_method2_additionalAU_alwaysOn(self,target,ref,v=False):

        n27=self.get_number_of_AUs_above_27(ref)
        return (len(set(target).intersection(set(ref)))+n27)/(len(ref)+n27)      

    def compute_matchScore(self,target,ref,method='method1',isBinary=False,BinaryTHRES=.7): 
        '''
            @param method
            method1: coderReliabilityMethod: jointly activated AU x 2 / all activated AU across target & reference pattern 
            method2: directMatchMethod: overlapped activated AU / activated AU of reference pattern     
            method1_sim_addAU: method1 + simulate activated AU above 27 (1000 simulations), randomly assigned whether AU above 27 within stereotype pattern is activated in target pattern
            method2_sim_addAU: method2 + simulate activated AU above 27 (1000 simulations), randomly assigned whether AU above 27 within stereotype pattern is activated in target pattern
            method1_allAddAU: method1 + assume AU above 27 always on
            method2_allAddAU: method2 + assume AU above 27 always on
        '''
        r=self.compute_match[method](target,ref)
        return (r>BinaryTHRES)*1 if(isBinary) else r
    
    def compute_median_matchScore_across_stereotypeVariants(self,target,refEmoCat,method):
        ''' 
            For a source where an emotion category does not have a Stereotype
            Assign that Emotion Category the value of -1 and return a matchscore of -1
            Filter this out in the final data since it will affec the median
        '''
        matchScores = list()
        for eachRef in self.STEREOTYPES[refEmoCat].values():
            if(eachRef[0]==-1):
                matchScores.append(-1)
            else:
                matchScores.append(self.compute_match[method](target,eachRef))
        return np.median(np.asarray(matchScores))

    def compute_median_matchScore_across_stereotypeVariants_addCordaroUSA(self,target,refEmoCat,method,BinaryTHRES=.7):
        matchScores = [self.compute_matchScore(target,eachRef,method) for eachRef in self.STEREOTYPES[refEmoCat].values()]

        from facsStereotypes import REF_AU_USA_VEC as CORDARO_USA_STEREOTYPES
        matchScores_Cordaro = [self.compute_matchScore(target,eachRef,method,isBinary=True,BinaryTHRES=BinaryTHRES) for eachRef in CORDARO_USA_STEREOTYPES[refEmoCat].values()]
        return np.median(np.asarray(matchScores+matchScores_Cordaro))

    def compute_max_matchScore_across_stereotypeVariants(self,target,refEmoCat,method):
        matchScores = [self.compute_match[method](target,eachRef) for eachRef in self.STEREOTYPES[refEmoCat].values()]
        return np.max(np.asarray(matchScores))


# In[5]:


class Analyzer():
    EMO_CAT=['Amusement', 'Anger', 'Awe', 'Contempt', 'Disgust', 'Embarrassment', 'Fear', 'Happiness', 'Interest', 'Pride', 'Sadness', 'Shame', 'Surprise']
    CATEGORY=['None','Weak','Moderate','Strong']
    AU_PATTERN_TYPES= {
            'original': 'refAU_13_ekman',
            'withICP' : 'refAU_13_ekman_plusICPonly',
            'ICP_and_USA' : 'refAU_13_Cordaro_ICPandUSA',
            'USAonly' : 'refAU_13_Cordaro_onlyUSA',
            'ICPalone': 'refAU_13_Cordaro_onlyICP',
        }
    TIE_BREAKING_METHODS = [
        'method1_highest_matchScore',
        'method2_mean_sd',
        'method3_randomly_selected'
    ]
    
    def __init__(self,
                 AU_PATTERN_TYPE=None,
                 METHOD=None,
                 RATING_INTENSITY=None,
                 TIE_BREAKER_MATCH_SCORE=None,
                 ACTIVATION_THRESHOLD=None,
                 SPECIFICITY_THRESHOLD=None,
                 USE_2nd_TOP_CATEGORY=None,
                 SPECIFICITY_categories=None):   
        self.__load_data(AU_PATTERN_TYPE,RATING_INTENSITY)   
        self.__initialize_vars_reliability(METHOD,USE_2nd_TOP_CATEGORY,TIE_BREAKER_MATCH_SCORE)
        self.__initialize_vars_specificity(ACTIVATION_THRESHOLD,SPECIFICITY_THRESHOLD,SPECIFICITY_categories)
        
        self.FLAG_PERFORMED={'reliability': False, 'specificity': False}
        
        self.check_and_perform_analysis = {
            'reliability': self.perform_reliability_analysis,
            'specificity': self.perform_specificity_analysis,
        }
    
    def check_and_perform_analysis_(self,TYPE='reliability'):
        if(self.FLAG_PERFORMED[TYPE] is not True):
            print('Performing analysis: ',TYPE)
            self.check_and_perform_analysis[TYPE]()
        else:
            print('...' + TYPE + ' analysis already performed. Skipping')
           
    def perform_specificity_analysis(self):
        dfSELECTED = self.dfAUlists[self.dfAUlists.index.isin(self.SELECTED)]
        dfs = self.__assign_poses_into_emotion_category_by_AUstereotypes_into_dict_of_dataframes(dfSELECTED,self.SPECIFICITY_THRESHOLD)        
        bool_Dfs=self.__assign_true_if_pose_specific_to_assignged_emotion_category_into_dict_of_booleanDataframes(dfs,self.dfMedian)

        
        df_stat=self.__compute_specificity_statistics_into_dataframe(bool_Dfs)
        
#         self.dfs = dfs
#         self.bool_Dfs = bool_Dfs
        self.FLAG_PERFORMED['specificity'] = True
        return df_stat
          
        
    def perform_reliability_analysis(self,USE_2nd_TOP_CATEGORY=None):        
        self.matchScores = self.__compute_604_times_13_matchscores(self.dfAUlists,self.METHOD,self.AU_PATTERN_TYPE,BinaryTHRES=1) 
        if(self.TIE_BREAKER_MATCH_SCORE):
            self.dfTopEmo = self.__get_topEmo()  
        else:
            self.dfTopEmo = self.__get_topEmo_basedOn_()
        
        self.USE_2nd_TOP_CATEGORY =  USE_2nd_TOP_CATEGORY if(USE_2nd_TOP_CATEGORY is not None) else self.USE_2nd_TOP_CATEGORY
        if(self.USE_2nd_TOP_CATEGORY): 
            print('USING 2nd TOP CATEGORY')
            self.dfTopEmo = self.__get_2ndtopEmo()
                
        self.reliabilityScore = self.__compute_reliability_score()
              
        self.FLAG_PERFORMED['reliability'] = True
       
    def __load_data(self,AU_PATTERN_TYPE,RATING_INTENSITY):     
        self.AU_PATTERN_TYPE = AU_PATTERN_TYPE or self.AU_PATTERN_TYPES['withICP']  
        self.matcher = Matcher(AU_PATTERN_FILE=self.AU_PATTERN_TYPE)
        
        self.dfAUlists = self.__load_AU_data()
        self.df_so = self.__load_rating_data()
        
        self.RATING_INTENSITY = RATING_INTENSITY or 0
        self.SELECTED=self.____select_scenarios_with_minimum_rating_intensity()   

    def __initialize_vars_reliability(self,METHOD,USE_2nd_TOP_CATEGORY,TIE_BREAKER_MATCH_SCORE):
        self.METHODS=['method1', 'method2', 'method1_sim_addAU', 'method2_sim_addAU', 'method1_allAddAU', 'method2_allAddAU']     
        self.METHOD = METHOD or 'method1_allAddAU'
        
        self.TIE_BREAKER_MATCH_SCORE = TIE_BREAKER_MATCH_SCORE or False
        
        self.USE_2nd_TOP_CATEGORY = USE_2nd_TOP_CATEGORY or False
        
        self.CORDARO_THRESHOLD = 0.7 #any values between (0,1)
    
    def __initialize_vars_specificity(self,ACTIVATION_THRESHOLD,SPECIFICITY_THRESHOLD,SPECIFICITY_categories):
        self.ACTIVATION_THRESHOLD=ACTIVATION_THRESHOLD or 0
        self.SPECIFICITY_THRESHOLD=SPECIFICITY_THRESHOLD or 0.4
        
        DEFAULT_SPECIFICITY_categories = ['Anger', 'Awe', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Interest', 'Sadness', 'Surprise']
        self.SPECIFICITY_categories = SPECIFICITY_categories if(SPECIFICITY_categories is not None) else self.EMO_CAT
          
        self.REF_AU_VEC = {emo : patterns for emo,patterns in self.matcher.STEREOTYPES.items() if emo in self.SPECIFICITY_categories}
        
        self.dfMedian = self.df_so.groupby('survey_questions_id').median()[self.EMO_CAT]
    
    ##=========================================================================
    def get_current_config(self):
        return {
            'METHOD': self.METHOD,
            'AU_PATTERN_TYPE': self.AU_PATTERN_TYPE,
            'RATING_INTENSITY': self.RATING_INTENSITY,
        }
        
        ##
        
    def count_topEmo_per_category(self):
        try:
            return self.dfTopEmo[0].value_counts().sort_index()
        except:
            return 'Top emo not defined'
    
    def plotConsistencyScore(self,y_axis='Score',ADD_OBSERVATION=False):           
        def label_standards():
            plt.title(y_axis)
            plt.xlabel('Emotion categories',fontsize=16,labelpad=20); plt.xticks(fontsize=16,rotation=70)
            plt.ylabel('Match score',fontsize=16)

        def plot_horizonal_dashed_lines():
            WEAK,MODERATE,STRONG=(0.2,0.4,0.7)
            for yPOS in (WEAK,MODERATE,STRONG): plt.hlines(yPOS,-.5,DASHED_WIDTH,linestyles='dashed')

        def label_consistency_on_the_right_margin():
            CONSISTENCY_TEXT_X=DASHED_WIDTH+.2
            consistencyLABELS= {'High':.85,'Moderate':.54,'Weak':.3,'None':.1}
            plt.text(CONSISTENCY_TEXT_X,1.05,'Consistency',fontweight='bold',fontsize=16)
            for LABEL,yPos in consistencyLABELS.items(): plt.text(CONSISTENCY_TEXT_X,yPos,LABEL,fontsize=16)

        def set_y_axis_range_to_see_whistles_at_0_and_1(): return plt.ylim([-0.1,1.02])

        ### MAIN PLOT   
        dataframe = self.reliabilityScore
        self.check_and_perform_analysis_('reliability')
        
        dataframe = self.reliabilityScore
        y_axis=dataframe.columns[1]
        
        EMO_LIST_EKMAN=[emo for emo in np.sort(dataframe['EmotionCategory'].unique())]

        DASHED_WIDTH=len(EMO_LIST_EKMAN)-.5
        plt.figure(figsize=(15,4))

        plot_horizonal_dashed_lines()

        sns.boxplot(x='EmotionCategory',y=y_axis,data=dataframe,
                   color='Grey',
                    order=EMO_LIST_EKMAN #plot according to alphabetical order
                   )

        label_standards()        
        label_consistency_on_the_right_margin()


        set_y_axis_range_to_see_whistles_at_0_and_1()
        
        return self.reliabilityScore.groupby('EmotionCategory').describe().T
        
    
    ##=========================================================================
    @staticmethod
    def __load_AU_data():   
        def ____represent_pose_with_ActionUnit_pattern_retrievedFrom(dfSource):
            AU_LIST=[str(i) for i in range(1,28)]   
            AU_LIST_and_ID=AU_LIST + ['id']
            df_AUs=dfSource[AU_LIST_and_ID].set_index('id')
            df_AUs.index.name='survey_questions_id'
            df_AUs=(df_AUs>=1)*1
            return df_AUs.sort_index()
        def transformAUpatternToAUlist(AUpattern): return [int(i) for i in AUpattern[AUpattern==1].index]
        df_scenes=pd.read_csv('../data/core_data/survey_questions.csv')
        df_AUs=____represent_pose_with_ActionUnit_pattern_retrievedFrom(df_scenes)
        psAUlists=df_AUs.apply(lambda AUpattern: transformAUpatternToAUlist(AUpattern),axis=1)
        dfAUlists=pd.DataFrame(psAUlists)
        return dfAUlists
    
    @staticmethod
    def __load_rating_data():
        def __get_individualRatings_from(dataframe,condition='/survey-so'): return dataframe[dataframe['survey_condition']==condition]
        
        raw_ratings = pd.read_csv('../data/core_data/survey_data.csv')
        return __get_individualRatings_from(raw_ratings)
        
    def __compute_604_times_13_matchscores(self,dfAUlists,METHOD,AU_PATTERN_TYPE,BinaryTHRES,v=True): 
        EMO_CAT = self.EMO_CAT
        
        dfT=dfAUlists.apply(
            lambda row: pd.Series([self.matcher.compute_median_matchScore_across_stereotypeVariants(
                    target=row[0],
                    refEmoCat=EMO,
                    method=METHOD,
                ) for EMO in EMO_CAT]),
            axis=1
        )
        dfT.columns=EMO_CAT
        dfT.index.name=METHOD+'_'+AU_PATTERN_TYPE
        return dfT
    
    ###    
    
    
    def __compute_reliability_score(self):
        r = pd.concat([self.dfTopEmo,self.matchScores],axis=1).apply(lambda row: pd.Series([row[0],row[row[0]]]),axis=1)
        r.columns=['EmotionCategory',self.METHOD]
        return r[r.index.isin(self.SELECTED)]
    
    def ____select_scenarios_with_minimum_rating_intensity(self):
        dfToKeep=self.df_so.groupby('survey_questions_id').median()[self.EMO_CAT].apply(lambda row: row.max()>=self.RATING_INTENSITY,axis=1)
        return dfToKeep[dfToKeep==True].index
    
    
    def __get_2ndtopEmo(self,EMO_CAT = EMO_CAT):
        def deepCopy_ensure_no_change_to(dfMedian): return pd.DataFrame.copy(dfMedian)
        def remove_top_emo_by_setting_ratingValue_to_zero(row,dfTopEmo): row[dfTopEmo.loc[row.name]]=0; return row
        def getMax(row):    return EMO_CAT[row[EMO_CAT].values.argmax()]

        return pd.DataFrame(
            deepCopy_ensure_no_change_to(self.dfMedian)\
                    .apply(lambda row: remove_top_emo_by_setting_ratingValue_to_zero(row,self.dfTopEmo),axis=1)\
                    .apply(lambda row: getMax(row),axis=1)
        )

        
    def __get_topEmo(self):     
        EMO_CAT = self.EMO_CAT
        def get_assignment_based_on_highest_match_score(combined,number=False):
            def getMax(row):
                if(len(row.Assignment_based_on_highest_ratings)>1):
                    if(sum(row[EMO_CAT].values==row[EMO_CAT].values.max())==1):
                        return EMO_CAT[row[EMO_CAT].values.argmax()]
                    else:
                        return EMO_CAT[row[EMO_CAT].values==row[EMO_CAT].values.max()] if number else '---TIED---'
                else:
                    return row.Assignment_based_on_highest_ratings[0]

            def extract_only_categories_tied_at_top(row): return row[row.Assignment_based_on_highest_ratings]

            return pd.concat([
                        combined['Assignment_based_on_highest_ratings'],
                        combined.apply(lambda row: extract_only_categories_tied_at_top(row),axis=1).fillna(0)],axis=1
                    )\
                         .apply(lambda row: getMax(row),axis=1)
        

        
        combined = pd.concat([self.____get_emoList_based_on_maxMedian_minIQR(self.df_so),self.matchScores],axis=1)
        
        topEmo = get_assignment_based_on_highest_match_score(combined)
        IND_TIED=topEmo[topEmo=='---TIED---'].index  
        df_TIES = self.____break_ties(IND_TIED)

        dfTopEmo=pd.DataFrame(topEmo)
        dfTopEmo.loc[IND_TIED]=df_TIES
        return dfTopEmo
    
    def __get_topEmo_basedOn_(self):     
        EMO_CAT = self.EMO_CAT
        def get_assignment_based_on_highest_match_score(combined,number=False):
            def getMax(row):
                if(len(row.Assignment_based_on_highest_ratings)>1):
                    if(sum(row[EMO_CAT].values==row[EMO_CAT].values.max())==1):
                        return EMO_CAT[row[EMO_CAT].values.argmax()]
                    else:
                        return EMO_CAT[row[EMO_CAT].values==row[EMO_CAT].values.max()] if number else '---TIED---'
                else:
                    return row.Assignment_based_on_highest_ratings[0]

            def extract_only_categories_tied_at_top(row): return row[row.Assignment_based_on_highest_ratings]

            return pd.concat([
                        combined['Assignment_based_on_highest_ratings'],
                        combined.apply(lambda row: extract_only_categories_tied_at_top(row),axis=1).fillna(0)],axis=1
                    )\
                         .apply(lambda row: getMax(row),axis=1)
        

        
        combined = pd.concat([self.____get_emoList_based_on_maxMedian_minIQR_maxMean_minStd(self.df_so),self.matchScores],axis=1)
        
        topEmo = get_assignment_based_on_highest_match_score(combined)
        IND_TIED=topEmo[topEmo=='---TIED---'].index  
        df_TIES = self.____break_ties(IND_TIED)

        dfTopEmo=pd.DataFrame(topEmo)
        dfTopEmo.loc[IND_TIED]=df_TIES
        return dfTopEmo
    
    def get_topEmo_randomly_when_tied(self):     
        return self.____get_emoList_based_on_maxMedian_minIQR(self.df_so)#             .applymap(lambda emoList: np.random.choice(emoList))
    
    def ____break_ties(self,IND_TIED): 
        return pd.DataFrame(
            self.____get_emoList_based_on_maxMedian_minIQR_maxMean_minStd(self.df_so[self.df_so.survey_questions_id.isin(IND_TIED)]).loc[IND_TIED]\
                .apply(lambda row: row.Assignment_based_on_highest_ratings[0],axis=1)
        )
    
    def ____get_emoList_based_on_maxMedian_minIQR(self,df_so):
        dat2={survey_questions_id : self.______getMinIQRCategories(df_so,survey_questions_id,self.______getMaxMedianCategories(df_so,survey_questions_id)) 
         for survey_questions_id in np.unique(df_so['survey_questions_id'].values)}

        dat22=pd.DataFrame(pd.Series(dat2))
        dat22.columns=['Assignment_based_on_highest_ratings']
        return dat22

    def ____get_emoList_based_on_maxMedian_minIQR_maxMean_minStd(self,df_so):
        dat2={survey_questions_id : 
              self.______getMinSTDCategories(df_so,survey_questions_id,
                  self.______getMaxMeanCategories(df_so,survey_questions_id,
                    self.______getMinIQRCategories(df_so,survey_questions_id,
                        self.______getMaxMedianCategories(df_so,survey_questions_id)))) 
         for survey_questions_id in np.unique(df_so['survey_questions_id'].values)}

        dat22=pd.DataFrame(pd.Series(dat2))
        dat22.columns=['Assignment_based_on_highest_ratings']
        return dat22
    
    def ____get_emoList_based_on_maxMedian_minIQR_maxMean(self,df_so):
        dat2={survey_questions_id : 
              self.______getMaxMeanCategories(df_so,survey_questions_id,
                self.______getMinIQRCategories(df_so,survey_questions_id,
                    self.______getMaxMedianCategories(df_so,survey_questions_id))) 
         for survey_questions_id in np.unique(df_so['survey_questions_id'].values)}

        dat22=pd.DataFrame(pd.Series(dat2))
        dat22.columns=['Assignment_based_on_highest_ratings']
        return dat22
    
    @staticmethod
    def ______getMaxMedianCategories(df,survey_questions_id,selectedCategory=EMO_CAT): 
        def get_maxMedian(df,survey_questions_id): return df[df.survey_questions_id==survey_questions_id][selectedCategory].median()==df[df.survey_questions_id==survey_questions_id][selectedCategory].median().max()

        maxMedian = get_maxMedian(df,survey_questions_id)
        return [i for i in maxMedian.index if maxMedian[i]]

    @staticmethod
    def ______getMinIQRCategories(df,survey_questions_id,selectedCategory=EMO_CAT):
        iqr = df[df.survey_questions_id==survey_questions_id][selectedCategory].apply(lambda col: np.quantile(col,.75)-np.quantile(col,.25),axis=0)
        minIQR = (iqr==iqr.min())
        return [i for i in minIQR.index if minIQR[i]]

    @staticmethod
    def ______getMaxMeanCategories(df,survey_questions_id,selectedCategory=EMO_CAT):
        def get_maxMean(df,survey_questions_id): return df[df.survey_questions_id==survey_questions_id][selectedCategory].mean()==df[df.survey_questions_id==survey_questions_id][selectedCategory].mean().max()

        maxMean = get_maxMean(df,survey_questions_id)
        return [i for i in maxMean.index if maxMean[i]]    

    @staticmethod
    def ______getMinSTDCategories(df,survey_questions_id,selectedCategory=EMO_CAT):
        std= df[df.survey_questions_id==survey_questions_id][selectedCategory].std()
        minStd=(std==std.min() )
        return [i for i in minStd.index if minStd[i]]
    
    
    ###      
    def generate_reliability_statistics_for_figure_3(self):
        self.check_and_perform_analysis_('reliability')
        return self.reliabilityScore.groupby('EmotionCategory')            .apply(lambda r: pd.Series(np.percentile(r,[5,50,95]),index=['p5','p50','p95'])).T
        
#     def generate_reliability_statistics_for_figure_3(self):
#         self.check_and_perform_analysis_('reliability')
#         return self.reliabilityScore.groupby('EmotionCategory').describe().T
    
    def generate_table_S7_reliabilityScore(self):
        self.check_and_perform_analysis_('reliability')
        
        def get_percent(row): return np.round(row.values/row.sum(),2)
        def get_total(row): return row.sum()
        
        t = self.__generate_table_count_consistency_category()
        index = [n+'(proportion)' for n in t.columns]

        
        def get_ordered_column(CAT): 
            l = []; 
            for i,c in enumerate(CAT): 
                l.append(c);l.append(index[i]); 
            return l
            
        df= pd.concat([
            t,
            t.apply(lambda row: pd.Series(get_percent(row),index=index),axis=1),
            t.apply(lambda row: pd.Series(get_total(row),index=['total']),axis=1),
        ],axis=1)
        
        return df[get_ordered_column(self.CATEGORY)]
        
    def __generate_table_count_consistency_category(self,CATEGORY=CATEGORY):
        g=self.reliabilityScore.set_index('EmotionCategory')            .applymap(lambda val: self.____assign_consistency_category(val))            .reset_index().groupby('EmotionCategory')
        return g[self.METHOD].value_counts().unstack().fillna(0).astype(int)[CATEGORY]
        
    @staticmethod
    def ____assign_consistency_category(val):
        if(val < 0.2): return 'None'
        elif(val < 0.4 and val >=0.2): return 'Weak'
        elif(val < 0.7 and val >= 0.4): return 'Moderate'
        elif(val >= 0.7): return 'Strong'
        else: return 'Error'

    def ____test_assign_consistency_category(self):
        assert(self.____assign_consistency_category(0.1)=='None')
        assert(self.____assign_consistency_category(0.2)=='Weak')
        assert(self.____assign_consistency_category(0.4)=='Moderate')
        assert(self.____assign_consistency_category(0.7)=='Strong')


    ##
    def generate_table_S8_BayesFactor(self):
        self.check_and_perform_analysis_('reliability')
        
        consistencyTabe = self.__generate_table_count_consistency_category()
        table=consistencyTabe.apply(lambda s: self.__compute_bayesFactor_and_significant_pairwise(s),axis=1)
        return table.applymap(lambda val: self.__formatTable(val))

    @staticmethod
    def __formatTable(data):
        SIGNIFICANT,BayesVal=(data['significant'],data['bayes_factor'])

        if(SIGNIFICANT):
            if(BayesVal>1000): return '>1000***'
            elif(BayesVal>100):return '{:.2f}***'.format(BayesVal)
            elif(BayesVal>30): return '{:.2f}**'.format(BayesVal)
            elif(BayesVal>10): return '{:.2f}*'.format(BayesVal)
            else:              return '{:.2f}'.format(BayesVal)
        else:
            if(BayesVal>1000): return '>1000'
            else:              return '{:.2f}'.format(BayesVal)

    def __compute_bayesFactor_and_significant_pairwise(self,s):
        n=s.sum()
        testResult=[];label=[]
        for i in range(0,len(s.index)):
            for j in range(i+1,len(s.index)):
                k1=s[i];k2=s[j];n1=n2=n
                testResult.append(self.____test_first_distributions_greater(k1,n1,k2,n2))
                label.append(s.index[i] + '>' + s.index[j])          
        return pd.Series(testResult,index=label)
    
    @staticmethod
    def ____test_first_distributions_greater(k1,n1,k2,n2,numSamp=10000,sigLevel=0.05,BayesFactorThres=0.5,v=False):
        '''
            Reference: Bayes factor ~ likelihood ratio http://statmath.wu.ac.at/research/talks/resources/talkheld.pdf
        '''
        def sample_betaDist(k,n,numSamp=1000,v=False): alpha=k+1;beta=n-k+1; return np.random.beta(alpha,beta,numSamp)
        def simulate_two_beta_distributions(k1,n1,k2,n2,numSamp=numSamp): return (sample_betaDist(k1,n1,numSamp),sample_betaDist(k2,n2,numSamp))      
        def compute_difference_between_a_pair_of_sample_from_each_distribution(s1,s2): return s1 - s2
        def compute_probability_s1_greater_s2(differenceDistribution,numSamp=numSamp): return np.sum(differenceDistribution>0)/numSamp  
        def compute_bayes_factor(s1,s2,DIV_BY_ZERO_CORRECTION = 1e-14): return np.sum(s1>BayesFactorThres)/(np.sum(s2>BayesFactorThres)+DIV_BY_ZERO_CORRECTION)    
        def compute_significant_either_direction(prob,sigLevel): return prob>(1-sigLevel) # or (prob < sigLevel)

        s1,s2 = simulate_two_beta_distributions(k1,n1,k2,n2) 
        sD=compute_difference_between_a_pair_of_sample_from_each_distribution(s1,s2); plt.hist(sD) if(v) else '';
        prob = compute_probability_s1_greater_s2(sD)

        return {
            'bayes_factor' : compute_bayes_factor(s1,s2),
            'prob'         : prob,
            'significant'  : compute_significant_either_direction(prob,sigLevel)
        }
    
    
    
    # ========== SPECIFICITY ANALYSIS 
    def __assign_poses_into_emotion_category_by_AUstereotypes_into_dict_of_dataframes(self,faceDataframe,SPECIFICITY_THRESHOLD):
        def retain_unique_set_of_matching_poses_by_removing_overlaps(inputDataframe):
            df = inputDataframe.drop_duplicates(subset=None, keep='first')
            return df.astype('int32')

        def combineDataframe(dataframe):
            dfTemp=pd.DataFrame()
            for key,value in dataframe.items():
                dfTemp=pd.concat([dfTemp,pd.Series(value.index)])  
            return dfTemp 

        dfU=dict()
        for emoCat,PATTERN in self.REF_AU_VEC.items():       
            dfVar = self.____pick_matching_poses_forEach_AUstereotypes_variants(faceDataframe,PATTERN,SPECIFICITY_THRESHOLD)  
            dfU[emoCat] = retain_unique_set_of_matching_poses_by_removing_overlaps(combineDataframe(dfVar))
            
        return dfU
    
    def ____pick_matching_poses_forEach_AUstereotypes_variants(self,faceAUsDataframe,EACH_STEREOTYPE,SPECIFICITY_THRESHOLD,v=False):
        dfT2=dict(); AU_PATTERN_COLUMN=0
        for index,VARIANT in EACH_STEREOTYPE.items():
            def compute_match_score_for_all_poses_against_variant(df,VARIANT): return df.apply(
                lambda AUpattern: self.matcher.compute_matchScore(AUpattern[AU_PATTERN_COLUMN],VARIANT,method=self.METHOD),axis=1)  
            def extractSubsetMeetingThreshold(pdSeries,SPECIFICITY_THRESHOLD): return pdSeries[pdSeries>=SPECIFICITY_THRESHOLD]

            print('iPattern,VARIATION_PATTERN =',VARIANT) if(v) else ''
            dfT1=compute_match_score_for_all_poses_against_variant(faceAUsDataframe,VARIANT)    
            dfT2[index]=extractSubsetMeetingThreshold(dfT1,SPECIFICITY_THRESHOLD)
        return dfT2
    

    def __assign_true_if_pose_specific_to_assignged_emotion_category_into_dict_of_booleanDataframes(self,dict_of_dataframes,dfMedianRating):
        d=dict(); dict_Dfs = dict_of_dataframes
        for emoCat in self.REF_AU_VEC.keys():
            d[emoCat]=dfMedianRating.loc[dict_Dfs[emoCat][0].values].apply(
                lambda row: self.____is_pose_NOT_specific_to_assigned_emotion_category(emoCat, row)*1,
                axis=1
            )
        return d
    
    @staticmethod
    def ____is_pose_NOT_specific_to_assigned_emotion_category(assignedCategory,poseMedianScenarioRatings):
        c,m=(assignedCategory,poseMedianScenarioRatings)
        def is_rating_of_assigned_emotion_category_BELOW(c,m,t): return poseMedianScenarioRatings[c] <= t
        def is_rating_of_another_emotion_category_ATLEAST(m,t): return sum(m>t) > 0
        return is_rating_of_assigned_emotion_category_BELOW(c,m,2) and is_rating_of_another_emotion_category_ATLEAST(m,2)
    
    def __compute_specificity_statistics_into_dataframe(self,dict_of_booleanDataframes):
        d=dict();bool_Dfs=dict_of_booleanDataframes
        for EMO,dfBool in bool_Dfs.items():
            returnNull = len(dfBool)==0
            k=dfBool.sum()
            n=dfBool.count()
            d[EMO]=self.____computeP_sig(n-k,n,returnNull=returnNull)
        return pd.DataFrame(d).T

    @staticmethod
    def ____computeP_sig(k,n,method='Bayesian',chance_level=0.167,lowerBound=True,returnNull=False):

        def compute_statistics_method_Bayesian(k,n):
            def gen_betaDist_samples(k,n,numSamp=1000,v=False): alpha=k+1;beta=n-k+1; return np.random.beta(alpha,beta,numSamp)
            def obtain_creditabilityInterval_bounds(): return tuple(np.percentile(samp,[2.5,97.5]))
            def obtain_mean_estimate(): return np.percentile(samp,50)

            samp=gen_betaDist_samples(k,n,numSamp=10000)
            p_hat=obtain_mean_estimate()
            CI_lower,CI_upper=obtain_creditabilityInterval_bounds()
            return p_hat,CI_lower,CI_upper

        def compute_statistics_method_Binomial(k,n):
            p_hat=k/n
            CI_dev=np.sqrt(p_hat*(1-p_hat)/n)
            CI_lower=(p_hat-CI_dev,p_hat+CI_dev)

        compute_statistics={
            'Bayesian': compute_statistics_method_Bayesian,
            'Binomial': compute_statistics_method_Binomial,
        }

        def computeP(k,n,method='Bayesian',v=False):
            ''' @param: method='Bayesian' or 'Binomial' '''
            p_hat,CI_lower,CI_upper=compute_statistics[method](k,n)
            return {'k':k,'n':n,'p_hat':round(p_hat,2),'CI_lower':round(CI_lower,2),'CI_upper':round(CI_upper,2)}


        def determineSignificance(CI_lower,chance_level=0.167): return CI_lower>chance_level

        ##
        if(returnNull): 
            return {'k': -1,'n':-1,'p_hat':-1,'CI_lower':-1,'CI_upper':-1,'significant':-1}
        else:
            result=computeP(k,n,method=method)
            result['significant']=determineSignificance(result['CI_lower'])*1
            return pd.Series(result) 


# In[6]:


class MultiverseProcessor():
    def __init__(self,SELECTED_FILES=None, ROOT_PATH='../data/multiverse/',
                 COLUMNS=[5,50,95],
                START_ROW=1,C_START=0,C_END=3):
        self.ROOT_PATH = ROOT_PATH
        self.FILES = SELECTED_FILES if(SELECTED_FILES is not None) else [f for f in os.listdir(ROOT_PATH)]
        self.COLUMNS = COLUMNS
        self.START_ROW=START_ROW
        self.C_START=C_START
        self.C_END=C_END
        
        self.d = self.combine_all_analysis_results()
#         self.data=self.calculate_all_statistics()
#         self.medianData = self.get_all_median_IQR()
     

    ### Reliability Multiverse 
    def calculate_all_statistics(self):
        data=dict()
        for p in [25,50,75]:
            data[p]=self.getFinalData_percentile(self.d,p)
            data[p]['score']=data[p]['Score'].astype(float)
        return data
    
    def get_all_median_IQR(self):
        def get_statistics(values,INVALID=-1): 
            median,l,u,p5,p95=np.percentile(values[values!=INVALID],[50,25,75])
            return median,l,u,u-l
        return self.d.apply(lambda row: pd.Series(get_statistics(row),index=['median','p25','p75','IQR']),axis=1)
    
    def get_and_reformat_table_from(self,FILE):
        t=pd.read_csv(self.ROOT_PATH+FILE).T.iloc[self.START_ROW:,self.C_START:self.C_END]
        t.columns=self.COLUMNS
        return t.unstack()

    def combine_all_analysis_results(self):
        d = pd.DataFrame()
        for f in self.FILES:
            ft=self.get_and_reformat_table_from(f)
            ft.name = f.split('_')[0]
            d = pd.concat([d,pd.DataFrame(ft)],axis=1)
        return d

    @staticmethod
    def getFinalData_percentile(d,percentile=50):
        dd=pd.DataFrame(d.loc[percentile].T.stack())
        ddd=dd.reset_index().drop('level_0',axis=1)
        ddd.columns = ['EmotionCategory','Score']
        ddd=ddd[ddd['Score']!=-1]
        return ddd
    
    ### Base rate calculation
    def compute_base_rate(plot=True):
        AUs=pd.read_csv('../data/for_import/dfAU.csv')

        AU_LIST = [str(i) for i in range(1,28)]
        AU_binary=AUs[AU_LIST].applymap(lambda val: int(val) > 0)

        totalScenarios = AU_binary.shape[0]
        return AU_binary.sum()/totalScenarios

    def plot_base_rate(baseRate):
        plt.figure(figsize=(10,5))
        plt.stem(baseRate.index,baseRate)
        plt.ylim([0,1])
        plt.xlabel('Action Unit (AU) number')
        plt.title('Base rate of Action Unit occurence among 604 faces')
        plt.show()

    def get_base_rate_statistics(baseRate):
        pd.DataFrame(baseRate[baseRate>0]).describe().loc[['min','50%','max']]


# In[7]:


class Plotter():        
    def plotConsistencyScore(self,dataframe,y_axis='Score',ADD_OBSERVATION=False):             
        return self.plot(dataframe,y_axis=y_axis,ADD_OBSERVATION=ADD_OBSERVATION)
    
    def plotSpecificityScore(self,dataframe,y_axis='p_hat',ADD_OBSERVATION=False):  
        return self.plot(dataframe,y_axis=y_axis,ADD_OBSERVATION=ADD_OBSERVATION,ANALYSIS_TYPE='Specificity')
    
    
    def plot(self,dataframe,y_axis='Score',x_axis='EmotionCategory',ADD_OBSERVATION=False,ANALYSIS_TYPE='Consistency'):           
        ### MAIN PLOT   
        
        EMO_LIST_EKMAN=[emo for emo in np.sort(dataframe[x_axis].unique())]

        DASHED_WIDTH=len(EMO_LIST_EKMAN)-.5
        plt.figure(figsize=(15,4))

        self.plot_horizonal_dashed_lines(DASHED_WIDTH)

        sns.boxplot(
            x=x_axis,
            y=y_axis,
            data=dataframe,
            color='Grey',
            order=EMO_LIST_EKMAN #plot according to alphabetical order
        )
        
        self.label_standards()        
        self.label_on_the_right_margin(DASHED_WIDTH,ANALYSIS_TYPE)


        self.set_y_axis_range_to_see_whistles_at_0_and_1()
        
        return dataframe.groupby(x_axis).describe().T
    
    @staticmethod
    def plot_boxplot(data):
        pass
    
    @staticmethod
    def plot_horizonal_dashed_lines(DASHED_WIDTH):
        WEAK,MODERATE,STRONG=(0.2,0.4,0.7)
        for yPOS in (WEAK,MODERATE,STRONG): plt.hlines(yPOS,-.5,DASHED_WIDTH,linestyles='dashed')

    @staticmethod
    def label_on_the_right_margin(DASHED_WIDTH,ANALYSIS_TYPE='Consistency'):
        CONSISTENCY_TEXT_X=DASHED_WIDTH+.2
        consistencyLABELS= {'High':.85,'Moderate':.54,'Weak':.3,'None':.1}
        plt.text(CONSISTENCY_TEXT_X,1.05,ANALYSIS_TYPE,fontweight='bold',fontsize=16)
        for LABEL,yPos in consistencyLABELS.items(): plt.text(CONSISTENCY_TEXT_X,yPos,LABEL,fontsize=16)
      
    @staticmethod
    def set_y_axis_range_to_see_whistles_at_0_and_1(): return plt.ylim([-0.1,1.02])
    
    @staticmethod
    def label_standards(TITLE='PlotTitle'):
            plt.title(TITLE)
            plt.xlabel('Emotion categories',fontsize=16,labelpad=20); plt.xticks(fontsize=16,rotation=70)
            plt.ylabel('Match score',fontsize=16)
            


# In[8]:


class SpecificationCurvePlotterSpecificity():
    SOURCES = [f for f in os.listdir('../data/for_import/ref_AU_auto/') if f.startswith('multiverse')]
    METHODS = ['method1', 'method2']
    RATING_INTENSITIES = [0,3]    
    
    EXPECTED_ARR_LEN = [len(val) for key,val in locals().items() if not key.startswith('_')] 
    
    def __init__(self,ROOT_PATH,EMOTION_CATEGORY='Amusement'):       

        bVAR_NAMES = self.get_booleanVar_Names(self.EXPECTED_ARR_LEN)
        for key,val in locals().items():
            if( key!= 'self'): setattr(self,key,val)  
        
        dfSpecificity = self.get_specificity_multiverse_data(ROOT_PATH)
        s=dfSpecificity.set_index(['EmotionCategory','id']).loc[EMOTION_CATEGORY].sort_values('p_hat')
        f=s['p_hat']
        f=pd.DataFrame(f)
        f.index.name = 'index'
        f=f.reset_index()
        dfF=pd.concat([
                f,
                f.apply(
                lambda row: self.decomposeToFactorVariables(row['index'],bVAR_NAMES,self.EXPECTED_ARR_LEN,EXPECTED_ID_LEN=3),axis=1
            )],axis=1
        )

        dfFF = dfF
        
        SHIFT=1
        dfDC=dfFF.set_index('index').apply(
            lambda row: pd.Series(row.values[SHIFT:]*np.arange(SHIFT,len(bVAR_NAMES)+SHIFT),bVAR_NAMES),axis=1
        ).T
        
        self.plot_specification_curve(s,dfDC,EMOTION_CATEGORY)
        
        for key,val in locals().items():
            if( key!= 'self'): setattr(self,key,val)  
    
    @staticmethod
    def get_booleanVar_Names(EXPECTED_ARR_LEN=EXPECTED_ARR_LEN,VAR_NAMES = ['p','m','r']):
        bVAR_NAMES=list()
        for index,varLen in enumerate(EXPECTED_ARR_LEN):
            for j in range(0,varLen):
                bVAR_NAMES.append(VAR_NAMES[index]+str(j))
        return bVAR_NAMES
    
    @staticmethod
    def decomposeToFactorVariables(ID,bVAR_NAMES,EXPECTED_ARR_LEN,EXPECTED_ID_LEN=3):
        if(len(ID)==EXPECTED_ID_LEN):
            D=dict()
            for i in range(0,EXPECTED_ID_LEN):
#                 print(ID,ID[i]) if(ID[i]=='5') else ''                
                t=np.zeros(EXPECTED_ARR_LEN[i])
                t[int(ID[i])]=True
                D[i]=t

            I = D[0]
            for i in range(1,EXPECTED_ID_LEN):
                I = np.append(I,D[i])
            return pd.Series(I,bVAR_NAMES)
        else:
            return pd.Series(np.ones(sum(EXPECTED_ARR_LEN))*-1,bVAR_NAMES)
        
    @staticmethod
    def plot_specification_curve(s,dfDC,EMOTION_CATEGORY):
        def plot_upper_figure(s,dfDC,EMOTION_CATEGORY):
            def label_standards():
                plt.title('Spread of median matchscore over multiverse analyses for each emotion category')
                plt.xlabel('Emotion categories',fontsize=16,labelpad=20); plt.xticks(fontsize=16,rotation=70)
                plt.ylabel('Match score',fontsize=16)

            def plot_horizonal_dashed_lines(DASHED_WIDTH):
                WEAK,MODERATE,STRONG=(0.2,0.4,0.7)
                for yPOS in (WEAK,MODERATE,STRONG): plt.hlines(yPOS,-.5,DASHED_WIDTH,linestyles='dashed')

            def label_consistency_on_the_right_margin(DASHED_WIDTH):
                CONSISTENCY_TEXT_X=DASHED_WIDTH+.2
                consistencyLABELS= {'High':.85,'Moderate':.54,'Weak':.3,'None':.1}
    #             plt.text(CONSISTENCY_TEXT_X-4,1.05,'Consistency',fontweight='bold',fontsize=16)
                for LABEL,yPos in consistencyLABELS.items(): plt.text(CONSISTENCY_TEXT_X+2,yPos,LABEL,fontsize=16,rotation=-90)

            def set_y_axis_range_to_see_whistles_at_0_and_1(): return plt.ylim([-0.1,1.02])
            
            def plot_main_part(s,dfDC,EMOTION_CATEGORY):
                plt.figure(figsize=(25,10))
                plt.title(EMOTION_CATEGORY,fontsize=24)
                plt.vlines(s.index,s['CI_lower'].values,s['CI_upper'].values,linewidth=1)
                plt.scatter(s.index,s['p_hat'].values,marker='o',color='.4')
                plt.ylim([0,1])
                plt.xticks(np.arange(0,len(dfDC.columns),10),np.arange(0,len(dfDC.columns),10),rotation='60',fontsize=12)
            
            plot_main_part(s,dfDC,EMOTION_CATEGORY)
            plot_horizonal_dashed_lines(DASHED_WIDTH=len(dfDC.columns))
            label_consistency_on_the_right_margin(DASHED_WIDTH=len(dfDC.columns))
            plt.show()

        def plot_lower_figure(dfDC):
            plt.figure(figsize=(25,10))
            for i in dfDC.index:
                plt.scatter(dfDC.columns,dfDC.loc[i])
            plt.yticks(np.arange(21)+1,dfDC.index)
            plt.hlines(0,0,len(dfDC.columns),color='white',linewidth=10)
            plt.xticks(np.arange(0,len(dfDC.columns),10),np.arange(0,len(dfDC.columns),10),rotation='60',fontsize=12)
            plt.show()
        
        plot_upper_figure(s,dfDC,EMOTION_CATEGORY)
        plot_lower_figure(dfDC)
        s
    ### Specificity Multiverse      
    def save_to_file_specificity_analysis_for_each_multiver_variant(self):
        for iS,SOURCE in enumerate(self.SOURCES):
            for iM,METHOD in enumerate(self.METHODS):
                for iR,RATING in enumerate(self.RATING_INTENSITIES):
                    ID = str(iS)+str(iM)+str(iR)
                    print(ID)
                    Analyzer(
                        AU_PATTERN_TYPE=SOURCE.split('.')[0],
                        METHOD=METHOD,
                        RATING_INTENSITY=RATING
                    ).perform_specificity_analysis()\
                    .to_csv('../data/multiverse_specificity/'+ ID +'_spec.csv')
                
    @staticmethod
    def get_specificity_multiverse_data(ROOT_PATH):
        def remove_invalid_data(df): return df[df.n!=-1]

        def combine_data_from_files(ROOT_PATH):   
            FILES = [f for f in os.listdir(ROOT_PATH)]

            df = pd.DataFrame()
            for f in FILES:
                dfR = pd.read_csv(ROOT_PATH+f)
                dfR['id'] = f.split('_')[0]
                df = pd.concat([df,dfR],sort=True)
            return df

        df=combine_data_from_files(ROOT_PATH)
        df=remove_invalid_data(df)
        df.columns=['CI_lower', 'CI_upper', 'EmotionCategory', 'id', 'k', 'n', 'p_hat','significant']
        df['specificity'] = df['p_hat'].apply(lambda val: 1-val)
        return df
        


# In[9]:


class SpecificationCurvePlotter():
    PATTERNS=[f for f in os.listdir('../data/for_import/ref_AU_auto/') if f.startswith('multiverse')]

    MATCH_SCORE_METHODS = [
        'method1','method2',
         'method1_sim_addAU','method2_sim_addAU',
         'method1_sim_addAU_baseline_max','method2_sim_addAU_baseline_max',
    ]

    TIE_BREAKING_METHOD = [False,True]
    RATING_INTENSITY = [0,3]
    SECOND_TOP_CATEGORY=[False,True]
    
    EXPECTED_ARR_LEN = [len(val) for key,val in locals().items() if not key.startswith('_')] 
    
    METHOD_INDEX=1
    METHOD_VARS=['M'+str(i) for i in range(0,EXPECTED_ARR_LEN[METHOD_INDEX])]
    
    def __init__(self,EMOCAT=None):
        self.EMOCAT = EMOCAT

    def compute_specification_curve_then_plot(self,ROOT_PATH = '../data/multiverse/',EMOTION_CATEGORY='Amusement'):
        
        EMOTION_CATEGORY = self.EMOCAT if(self.EMOCAT is not None) else EMOTION_CATEGORY
        
        bVAR_NAMES = self.get_booleanVar_Names(self.EXPECTED_ARR_LEN)
        
        m = MultiverseProcessor(ROOT_PATH=ROOT_PATH)
        s=self.get_all_s(m.d,EMOTION_CATEGORY=EMOTION_CATEGORY)
        f=s[50].reset_index()
        dfF=pd.concat([
                f,
                f.apply(
                lambda row: self.decomposeToFactorVariables(row['index'],bVAR_NAMES,self.EXPECTED_ARR_LEN),axis=1
            )],axis=1
        )
        dfFF=dfF[dfF['p0']!=-1]
        
        dfFFF=pd.concat([
            dfFF,
            dfFF[self.METHOD_VARS].apply(
            lambda row: self.transform_method_M_into_2_separate_variables(row),axis=1
            )        
        ],axis=1
        ).drop(self.METHOD_VARS,axis=1)
        
        bVAR_NAMES_final = dfFFF.set_index('index').columns[1:]

        SHIFT=1
        dfDC=dfFFF.set_index('index').apply(
            lambda row: pd.Series(row.values[SHIFT:]*np.arange(SHIFT,len(bVAR_NAMES_final)+SHIFT),bVAR_NAMES_final),axis=1
        ).T
        
        self.plot_specification_curve(s,dfDC,EMOTION_CATEGORY)

        for key,val in locals().items():
            if( key!= 'self'): setattr(self,key,val)        

                
    ####
    
    def get_all_items_to_analyse(self,v=False):
        count=0
        params=dict()
        for iP,pattern in enumerate(self.PATTERNS):
            for iM,method in enumerate(self.MATCH_SCORE_METHODS):
                for iR,ratingIntensity in enumerate(self.RATING_INTENSITY):
                    for iT,TieBreakingMethod in enumerate(self.TIE_BREAKING_METHOD):
                        for i2, USE_2ndTop in enumerate(self.SECOND_TOP_CATEGORY):
                            ID = str(iP)+str(iM)+str(iR)+str(iT)+str(i2)
                            count +=1
                            print(count,ID,' pattern =',pattern[:-4],' method =',method,
                                  ' ratingIntensity =',ratingIntensity,'TieBreakMatchScore =',TieBreakingMethod,
                                 ' use2ndTop =',USE_2ndTop
                                 ) if(v) else ''
                            params[ID] = {
                                    'pattern':pattern[:-4],
                                    'method':method,
                                    'ratingIntensity':ratingIntensity,
                                    'TieBreakMatchScore':TieBreakingMethod,
                                    'use2ndTop':USE_2ndTop
                                }

        return params

    
    def save_each_itemsToAnalyzelysis_to_csv(self,itemsToAnalyze,ROOT_PATH = '../data/multiverse/'):
        def perform_analysis(key,params,ROOT_PATH):
            a = Analyzer(
                AU_PATTERN_TYPE=params['pattern'],
                METHOD=params['method'],
                RATING_INTENSITY=params['ratingIntensity'],
                TIE_BREAKER_MATCH_SCORE=params['TieBreakMatchScore'],
                USE_2nd_TOP_CATEGORY=params['use2ndTop']
            )
            stats=a.generate_reliability_statistics_for_figure_3()
            stats.to_csv(ROOT_PATH+str(key)+'_re.csv')
            return stats  

        IDs_processed = [f.split('_')[0] for f in os.listdir(ROOT_PATH)]
        print('Total num analysis =',len(itemsToAnalyze.keys()))
        print('Num processed analyses =',len(IDs_processed))
        print('Start time is Singapore time ',datetime.now())

        countNoKey=0
        for key,params in itemsToAnalyze.items():
            if(str(key) not in IDs_processed):
                countNoKey+=1
                perform_analysis(key,params,ROOT_PATH=ROOT_PATH)
                print(countNoKey,key)
        print('End time is Singapore time ',datetime.now())

    def operate_on_files():
        import subprocess as sp
        ROOT_PATH_WIN = "..\\data\\multiverse\\" #Note: subprocess on Windows will open Windows cmd -> so need to use window format

        for index,file in enumerate(SELECTED_FILES):
            pass
        #     print(file, ' to ', TO_RENAME_TO[index],end=' : ')
        #     print(index,sp.call(['del',ROOT_PATH_WIN+file],shell=True))
        #     print(sp.call(['rename',ROOT_PATH_WIN+fileToRename,TO_RENAME_TO[index]],shell=True))
    
    
    
    ####
                
    @staticmethod
    def get_all_s(data,EMOTION_CATEGORY='Amusement',COLUMNS=[5,50,95]):
        s=dict()
        def get_sorted_median(dfMedian):
            a=dfMedian.loc[EMOTION_CATEGORY]
            return a[a>0].sort_values()
        
        median = get_sorted_median(data.loc[50])
        INDEX = median.index 
        
        s[5] = data.loc[5].loc[EMOTION_CATEGORY][INDEX]
        s[95]= data.loc[95].loc[EMOTION_CATEGORY][INDEX]
        
        s[50] = median
        
        return s

    @staticmethod
    def get_booleanVar_Names(EXPECTED_ARR_LEN=EXPECTED_ARR_LEN,VAR_NAMES = ['p','M','r','b','t']):
        bVAR_NAMES=list()
        for index,varLen in enumerate(EXPECTED_ARR_LEN):
            for j in range(0,varLen):
                bVAR_NAMES.append(VAR_NAMES[index]+str(j))
        return bVAR_NAMES

    @staticmethod
    def decomposeToFactorVariables(ID,bVAR_NAMES,EXPECTED_ARR_LEN,EXPECTED_ID_LEN=5):
        if(len(ID)==EXPECTED_ID_LEN):
            D=dict()
            for i in range(0,EXPECTED_ID_LEN):
    #             print(ID,ID[i]) if(ID[i]=='5') else ''
                t=np.zeros(EXPECTED_ARR_LEN[i])
                t[int(ID[i])]=True
                D[i]=t

            I = D[0]
            for i in range(1,EXPECTED_ID_LEN):
                I = np.append(I,D[i])
            return pd.Series(I,bVAR_NAMES)
        else:
            return pd.Series(np.ones(sum(EXPECTED_ARR_LEN))*-1,bVAR_NAMES)

    @staticmethod
    def transform_method_M_into_2_separate_variables(arr):
        '''
            input: values of
                - M variables: M0, M1, M2, M3, M4, M5
            output: values of 
                - m variable: 
                    M0, M2, M4 -> m0
                    M1, M3, M5 -> m1
                - s variable:
                    M0, M1 -> s0
                    M2, M3 -> s1
                    M4, M5 -> s2

        '''
        def transform_M_into_m(arr):
            m0=arr.M0 or arr.M2 or arr.M4
            m1=arr.M1 or arr.M3 or arr.M5
            return pd.Series([m0,m1],index = ['m0','m1'])

        def transform_M_into_s(arr):
            s0=arr.M0 or arr.M1
            s1=arr.M2 or arr.M3
            s2=arr.M4 or arr.M5
            return pd.Series([s0,s1,s2],index = ['s0','s1','s2'])

        return transform_M_into_m(arr)            .append(transform_M_into_s(arr))

    @staticmethod
    def plot_specification_curve(s,dfDC,EMOTION_CATEGORY):
        def plot_upper_figure(s,dfDC,EMOTION_CATEGORY):
            def label_standards():
                plt.title('Spread of median matchscore over multiverse analyses for each emotion category')
                plt.xlabel('Emotion categories',fontsize=16,labelpad=20); plt.xticks(fontsize=16,rotation=70)
                plt.ylabel('Match score',fontsize=16)

            def plot_horizonal_dashed_lines(DASHED_WIDTH):
                WEAK,MODERATE,STRONG=(0.2,0.4,0.7)
                for yPOS in (WEAK,MODERATE,STRONG): plt.hlines(yPOS,-.5,DASHED_WIDTH,linestyles='dashed')

            def label_consistency_on_the_right_margin(DASHED_WIDTH):
                CONSISTENCY_TEXT_X=DASHED_WIDTH+.2
                consistencyLABELS= {'High':.85,'Moderate':.54,'Weak':.3,'None':.1}
    #             plt.text(CONSISTENCY_TEXT_X-4,1.05,'Consistency',fontweight='bold',fontsize=16)
                for LABEL,yPos in consistencyLABELS.items(): plt.text(CONSISTENCY_TEXT_X+2,yPos,LABEL,fontsize=16,rotation=-90)

            def set_y_axis_range_to_see_whistles_at_0_and_1(): return plt.ylim([-0.1,1.02])
            
            def plot_main_part(s,dfDC,EMOTION_CATEGORY):
                plt.figure(figsize=(25,10))
                plt.title(EMOTION_CATEGORY,fontsize=24)
                plt.vlines(s[50].index,s[5].values,s[95].values,linewidth=1)
                plt.scatter(s[50].index,s[50].values,marker='o',color='.4')
                plt.ylim([0,1])
                plt.xticks(np.arange(0,len(dfDC.columns),10),np.arange(0,len(dfDC.columns),10),rotation='60',fontsize=12)
            
            plot_main_part(s,dfDC,EMOTION_CATEGORY)
            plot_horizonal_dashed_lines(DASHED_WIDTH=len(dfDC.columns))
            label_consistency_on_the_right_margin(DASHED_WIDTH=len(dfDC.columns))
            plt.show()

        def plot_lower_figure(dfDC):
            plt.figure(figsize=(25,10))
            for i in dfDC.index:
                plt.scatter(dfDC.columns,dfDC.loc[i])
            plt.yticks(np.arange(21)+1,dfDC.index)
            plt.hlines(0,-2,len(dfDC.columns),color='white',linewidth=10)
            plt.xticks(np.arange(0,len(dfDC.columns),10),np.arange(0,len(dfDC.columns),10),rotation='60',fontsize=12)
            plt.show()
        
        plot_upper_figure(s,dfDC,EMOTION_CATEGORY)
        plot_lower_figure(dfDC)
    


# In[10]:


m = MultiverseProcessor(ROOT_PATH='../data/multiverse/',COLUMNS=[25,50,75],START_ROW=2,C_START=4,C_END=7)


# In[11]:


m.calculate_all_statistics()[50]


# In[12]:


Plotter().plotConsistencyScore(m.calculate_all_statistics()[50],y_axis='score')


# In[13]:


s= SpecificationCurvePlotterSpecificity(ROOT_PATH='../data/multiverse_specificity/')


# In[14]:


s.dfSpecificity.EmotionCategory.unique()


# In[15]:


SELECTED=['Anger', 'Awe', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Interest']


# In[16]:


dfS=s.dfSpecificity[s.dfSpecificity.EmotionCategory.isin(SELECTED)]


# In[17]:


Plotter().plotSpecificityScore(dfS,y_axis='p_hat')


# In[18]:


a = Analyzer()


# In[19]:


a.EMO_CAT


# In[20]:


s = SpecificationCurvePlotter()


# In[ ]:


for EMO in a.EMO_CAT:
    s.compute_specification_curve_then_plot(ROOT_PATH='../data/multiverse_CI/',EMOTION_CATEGORY=EMO)


# In[22]:


for EMO in a.EMO_CAT:
    SpecificationCurvePlotterSpecificity(EMOTION_CATEGORY=EMO,ROOT_PATH='../data/multiverse_specificity/')

