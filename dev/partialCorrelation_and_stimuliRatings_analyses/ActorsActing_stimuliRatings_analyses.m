clear;
clc;

%% load and clean data
surveyData = importdata('survey_data.csv');
questionID = str2double(surveyData.textdata(2:end,1));
condition = str2double(replace(surveyData.textdata(2:end,3),{'ScenarioAlone','FaceAlone','FaceScenario'},{'1','2','3'}));
emotions = surveyData.textdata(1,4:end);
ratings = array2table(surveyData.data,'VariableNames',emotions);
sortedEmotions = sort(ratings.Properties.VariableNames);
sortedRatings = ratings(:,sortedEmotions);
allData = [questionID condition table2array(sortedRatings)];
allData = sortrows(allData,1);
% partition ratings by condition
scenarioAlone = allData(allData(:,2)==1,:); % 1 = scenario alone ratings
faceAlone = allData(allData(:,2)==2,:); % 2 = face alone ratings
scenarioFace = allData(allData(:,2)==3,:); % 3 = scenario+face ratings
stimuli = unique(allData(:,1));

%% get medians, IQRs, means, and SDs per stimulus, per condition
for i_stim = 1:length(stimuli)
    scenarioAlone_medians(i_stim,:) = median(scenarioAlone(scenarioAlone(:,1)==i_stim,3:end));
    scenarioAlone_IQRs(i_stim,:) = iqr(scenarioAlone(scenarioAlone(:,1)==i_stim,3:end));
    scenarioAlone_means(i_stim,:) = mean(scenarioAlone(scenarioAlone(:,1)==i_stim,3:end));
    scenarioAlone_SDs(i_stim,:) = std(scenarioAlone(scenarioAlone(:,1)==i_stim,3:end));
    faceAlone_medians(i_stim,:) = median(faceAlone(faceAlone(:,1)==i_stim,3:end));
    faceAlone_IQRs(i_stim,:) = iqr(faceAlone(faceAlone(:,1)==i_stim,3:end));
    faceAlone_means(i_stim,:) = mean(faceAlone(faceAlone(:,1)==i_stim,3:end));
    faceAlone_SDs(i_stim,:) = std(faceAlone(faceAlone(:,1)==i_stim,3:end));
    scenarioFace_medians(i_stim,:) = median(scenarioFace(scenarioFace(:,1)==i_stim,3:end));
    scenarioFace_IQRs(i_stim,:) = iqr(scenarioFace(scenarioFace(:,1)==i_stim,3:end));
    scenarioFace_means(i_stim,:) = mean(scenarioFace(scenarioFace(:,1)==i_stim,3:end));
    scenarioFace_SDs(i_stim,:) = std(scenarioFace(scenarioFace(:,1)==i_stim,3:end));
end

%% assign stimuli to emotion categories
% import assignment based on max median > min IQR > max match score
assignmentData = importdata('assignment_data.csv');
assignedCategory = assignmentData.data(:,2);
assignedCategory_Table = splitvars(table(tabulate(assignedCategory)),'Var1','NewVariableNames',{'EmotionCategory','Count','Percent'});
    
%% run one-way ANOVAs with planned contrasts to confirm category assignments
numCats = numel(unique(assignedCategory));
for i_cat = 1:numCats
    toTest = scenarioAlone_medians(assignedCategory==i_cat,:);
    meanIntensity_assigned(i_cat) = mean(toTest(:,i_cat));
    sdIntensity_assigned(i_cat) = std(toTest(:,i_cat));
    meanIntensity_all(i_cat,:) = mean(toTest);
    sdIntensity_all(i_cat,:) = std(toTest);
    [F_p(i_cat),tbl,stats] = anova1(toTest,[],'off');
    F(i_cat) = tbl{2,5};
    contrastCoefs = -1*ones(1,numCats);
    contrastCoefs(i_cat) = 12;
    contrast(i_cat) = sum(stats.means.*contrastCoefs);
    t(i_cat) = contrast(i_cat)/sqrt(tbl{3,4}*sum((contrastCoefs.^2/size(toTest,1))));
    df(i_cat) = tbl{3,3};
    t_p(i_cat) = 1-tcdf(t(i_cat),df(i_cat));
    clear toTest tbl stats
end
% create summary table (i.e., Table 1)
ANOVAsummary = [meanIntensity_assigned' sdIntensity_assigned' F' F_p' df' contrast' t' t_p'];
ANOVAsummary_Table = array2table(ANOVAsummary);
ANOVAsummary_Table = horzcat(assignedCategory_Table(:,2),ANOVAsummary_Table);
ANOVAsummary_Table.Properties.RowNames = {'Amusement','Anger','Awe','Contempt','Disgust','Embarrassment','Fear','Happiness','Interest','Pride','Sadness','Shame','Surprise'};
ANOVAsummary_Table.Properties.VariableNames = {'numScenarios','meanIntensity','sdIntensity','F','F_p','df','Contrast','t','t_p'};

%% identify high-intensity stimuli as scenario-alone rating for assigned category >= 3
highIntensity = zeros(length(stimuli),1);
for i_stim = 1:length(stimuli)
    if scenarioAlone_medians(i_stim,assignedCategory(i_stim)) >= 3
        highIntensity(i_stim) = 1;
    end
end
assignedCategory_HI = assignedCategory(highIntensity==1);
assignedCategory_HI_Table = splitvars(table(tabulate(assignedCategory_HI)),'Var1','NewVariableNames',{'EmotionCategory','Count','Percent'});
% get descriptive statistics for intensity ratings per category
scenarioAlone_medians_HI = scenarioAlone_medians(highIntensity==1,:);
for i_cat = 1:numCats
    toTest = scenarioAlone_medians_HI(assignedCategory_HI==i_cat,:);
    meanIntensity_all_HI(i_cat,:) = mean(toTest,1);
    sdIntensity_all_HI(i_cat,:) = std(toTest,1);
    clear toTest
end

%% compute complexity for all conditions
for i_stim = 1:length(stimuli)
    scenarioAlone_complexity(i_stim) = sum(scenarioAlone_medians(i_stim,:)>=1)/numCats;
    faceAlone_complexity(i_stim) = sum(faceAlone_medians(i_stim,:)>=1)/numCats;
    scenarioFace_complexity(i_stim) = sum(scenarioFace_medians(i_stim,:)>=1)/numCats;
end
% compare complexity for scenario alone vs. face alone ratings
[~,p,ci,stats] = ttest2(scenarioAlone_complexity,faceAlone_complexity); % use ttest for paired samples
d = computeCohen_d(scenarioAlone_complexity,faceAlone_complexity,'independent'); % option to change 3rd paramter to 'paired'
complexitySummary = [mean(scenarioAlone_complexity) std(scenarioAlone_complexity) mean(faceAlone_complexity) std(faceAlone_complexity)...
    stats.tstat stats.df p ci(1) ci(2) d];
complexitySummary_Table = array2table(complexitySummary);
complexitySummary_Table.Properties.VariableNames = {'mean_SA','SD_SA','mean_FA','SD_FA','t','df','p','CI_l','CI_u','d'};
% compare complexity for low vs. high intensity scenarios
[~,p_4,ci_4,stats_4] = ttest2(scenarioAlone_complexity(highIntensity==1),scenarioAlone_complexity(highIntensity==0));
d_4 = computeCohen_d(scenarioAlone_complexity(highIntensity==1),scenarioAlone_complexity(highIntensity==0));
intensitySummary = [sum(highIntensity==1) mean(scenarioAlone_complexity(highIntensity==1)) std(scenarioAlone_complexity(highIntensity==1)) ...
    sum(highIntensity==0) mean(scenarioAlone_complexity(highIntensity==0)) std(scenarioAlone_complexity(highIntensity==0)) ...
    stats_4.tstat stats_4.df p_4 ci_4(1) ci_4(2) d_4];
intensitySummary_Table = array2table(intensitySummary);
intensitySummary_Table.Properties.VariableNames = {'n_high','mean_high','SD_high','n_low','mean_low','SD_low','t','df','p','CI_l','CI_u','d'};

%% compute false positive rates based on face alone ratings
for i_cat = 1:numCats
    totalRated(i_cat) = sum(faceAlone_medians(:,i_cat)>=1);
    assignedRated(i_cat) = sum(faceAlone_medians(assignedCategory==i_cat,i_cat)>=1);
    falsePositiveRate(i_cat) = (totalRated(i_cat)-assignedRated(i_cat))/totalRated(i_cat);
end
% repeat for facial poses associated with high-intensity scenarios
faceAlone_medians_HI = faceAlone_medians(highIntensity==1,:);
for i_cat = 1:numCats
    totalRated_HI(i_cat) = sum(faceAlone_medians_HI(:,i_cat)>=1);
    assignedRated_HI(i_cat) = sum(faceAlone_medians_HI(assignedCategory_HI==i_cat,i_cat)>=1);
    falsePositiveRate_HI(i_cat) = (totalRated_HI(i_cat)-assignedRated_HI(i_cat))/totalRated_HI(i_cat);
end
    


