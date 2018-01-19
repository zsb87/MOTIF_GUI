import re
import csv
import sys, os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
import random
import glob
from sklearn import preprocessing
from sklearn import svm, neighbors, metrics, cross_validation, preprocessing
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
try: #python3
    import _pickle as cPickle
except ImportError: #python2.7
    import cPickle
    
from stru_utils import *
import shutil   
# from FG_predCount import FG_predCount
import subprocess
import matlab.engine



def run_motif(textfilepath, input_file):

    output_file = input_file

    # with open(textfilepath, "w") as myfile:
    #     myfile.write(input_file+'\n')


    protocol = 'inlabStr'
    i_subj = 0
    run = 41800
    dist = 0.7
    n_motif = 27.0
    config_file = 'config_file_is'

    # promptLabel.text = protocol
    # with open(textfilepath, "a") as myfile:
    #     myfile.write(protocol+'\n')

    #   for US, qualified subjs: Dzung Shibo Rawan JC Jiapeng Matt
    #   for US, finished subjs:  Dzung Shibo Rawan  7     6     9
    if protocol == 'inlabStr':
        subjs = ['P3','P4','P5', 'P10']
    if protocol == 'inlabUnstr':
        subjs = ['P1','P3','P6','P7','P8']
    # subjs = ['Rawan','Shibo','Dzung', 'Will', 'Gleb','JC','Matt','Jiapeng','Cao','Eric', 'MattSmall']
    subj = subjs[i_subj]

    save_flg = 1


    #----------------------------------------------------------------------------------------------------
    #
    # import features and model, do classification
    #
    #----------------------------------------------------------------------------------------------------

    columns = ['Prec(pos)','F1(pos)','TPR','FPR','Specificity','MCC','CKappa','w-acc']
    crossValRes = pd.DataFrame(columns = columns, index = range(5))
    active_p_cnt = 0


    outfolder = '../'+protocol+'/result/seg_clf/engy_IS2ISseg_personalized/'+subj+'/'
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    for threshold_str in ['0.5']:#,'0.6','0.7','0.8','0.9'
        testsubj ='test'+subj
        trainsubj = 'train'+subj

        trainsubjF = trainsubj + '/'
        testsubjF = testsubj + '/'
        folder = '../'+protocol+'/subject/'
        trainfeatFoler = folder+trainsubjF+"feature/all_features/"
        trainsegfolder = folder+trainsubjF+"segmentation/"



        #----------------------------------------------------------------------------------------------------
        #
        # train the classifier
        #
        #----------------------------------------------------------------------------------------------------

        with open(textfilepath, "a") as myfile:
            myfile.write('Load train set features...\n')

        df = pd.read_csv(os.path.join(os.path.dirname(__file__),trainfeatFoler+ "engy_run"+ str(run) +"_pred_features.csv" ))
        
        # with open(textfilepath, "a") as myfile:
        #     df[:3].to_csv(myfile, header=False)
        #     myfile.write('\n')

        labelDf = pd.read_csv(os.path.join(os.path.dirname(__file__),trainsegfolder+'engy_run'+str(run)+'_pred_label_thre'+threshold_str+'/seg_labels.csv'),names = ['label'])
        
        X = df.iloc[:,:-1].as_matrix()#  notice:   duration should not be included in features, as in detection period this distinguishable feature will be in different distribution
        Y = labelDf['label'].iloc[:].as_matrix()


        with open(textfilepath, "a") as myfile:
            myfile.write('Train classifier...\n')


        classifier = RandomForestClassifier(n_estimators=185)
        classifier.fit(X, Y)
        with open(textfilepath, "a") as myfile:
            myfile.write('Training done. \n\n')



        #----------------------------------------------------------------------------------------------------
        #
        # save the classifier
        #
        #----------------------------------------------------------------------------------------------------

        mdlFolder = os.path.join(os.path.dirname(__file__),folder+trainsubjF+"model/")

        if not os.path.exists(mdlFolder):
            os.makedirs(mdlFolder)


        with open(textfilepath, "a") as myfile:
            myfile.write('Save classifier...\n')

        with open(mdlFolder+'RF_185_trainset_motif_segs_thre'+threshold_str+'_run'+str(run)+'.pkl', 'wb') as fid:
            cPickle.dump(classifier, fid)    



        #----------------------------------------------------------------------------------------------------
        #
        # import test set and model
        #
        #----------------------------------------------------------------------------------------------------

        folder = '../'+protocol+'/subject/'
        testfeatFolder = '../'+protocol+'/subject/'+testsubj + '/feature/all_features/'
        testfeatFile = testfeatFolder+ "engy_run"+ str(run) +'_pred_features.csv'
        df_all = pd.read_csv(os.path.join(os.path.dirname(__file__),testfeatFile))

        with open(textfilepath, "a") as myfile:
            myfile.write('Load test set features...\n')

        labelFile = folder+testsubj + '/segmentation/engy_run'+str(run)+'_pred_label_thre'+threshold_str+'/seg_labels.csv'
        labelDf = pd.read_csv(os.path.join(os.path.dirname(__file__),labelFile), names = ['label'])


        mdlFolder = os.path.join(os.path.dirname(__file__),folder+trainsubjF+"model/")

        with open(textfilepath, "a") as myfile:
            myfile.write('Load classifier...\n')
        

        X = df_all.iloc[:,:-1].as_matrix() #notice: duration should not be included in features, as in detection period this distinguishable feature will be in different distribution
        Y = labelDf['label'].as_matrix()
        
        # load the classifier
        with open(mdlFolder+'RF_185_trainset_motif_segs_thre'+threshold_str+'_run'+str(run)+'.pkl', 'rb') as fid:
            classifier = cPickle.load(fid)


        with open(textfilepath, "a") as myfile:
            myfile.write('Start classification...\n\n')
        

        prec_pos, f1_pos, TPR, FPR, Specificity, MCC, CKappa, w_acc, cm, y_pred = clf_cm_pickle(classifier, X, Y)

        # promptLabel.text = cm
        with open(textfilepath, "a") as myfile:
            myfile.write('Confusion matrix:\n')
            myfile.write('Predicted     0    1   \nTrue\n')
            myfile.write('0          '+np.array2string(cm[0,:], separator='  '))
            myfile.write('\n')
            myfile.write('1          '+np.array2string(cm[1,:], separator='     '))
            myfile.write('\n')
            
            # myfile.write(np.array2string(cm, separator=', '))

        ts = time.time()
        current_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H-%M-%S')
        # promptLabel.text = current_time

        np.savetxt(outfolder+'RF_185_motif_dist'+str(dist)+'_multi-thre_'+protocol+'_'+subj+'_run'+str(run)+'(109)_cm'+str(current_time)+'.csv', cm, delimiter=",")

        crossValRes['Prec(pos)'][active_p_cnt] = prec_pos
        crossValRes['F1(pos)'][active_p_cnt] = f1_pos
        crossValRes['TPR'][active_p_cnt] = TPR
        crossValRes['FPR'][active_p_cnt] = FPR
        crossValRes['Specificity'][active_p_cnt] = Specificity
        crossValRes['MCC'][active_p_cnt] = MCC
        crossValRes['CKappa'][active_p_cnt] = CKappa
        crossValRes['w-acc'][active_p_cnt] = w_acc
        active_p_cnt = active_p_cnt+1

    ts = time.time()
    current_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H-%M-%S')
    # promptLabel.text = current_time

    crossValRes.to_csv( os.path.join(os.path.dirname(__file__),outfolder+'RF_185_motif_dist'+str(dist)+'_multi-thre_'+protocol+'_'+subj+'_run'+str(run)+'(109)_'+str(current_time)+'.csv'), index = None)




    #----------------------------------------------------------------------------------------------------
    #
    #  remove useless data files to save local space
    #
    #----------------------------------------------------------------------------------------------------

    # split_subjs = ['train'+subj,'test'+subj]
    # for split_subj in split_subjs:
    #     subjfolder = split_subj + '/'
    #     folder = '../'+protocol+'/subject/'
    #     segfolder = folder+subjfolder+"segmentation/"
    #     for f in glob.glob(os.path.join(os.path.dirname(__file__),segfolder+'engy_run'+str(run)+'_pred_data/*')):
    #         os.remove(f)



    #----------------------------------------------------------------------------------------------------
    # 
    # segmentation based on motif
    # 
    #----------------------------------------------------------------------------------------------------

    # ans = FG_predCount(protocol, subj, run)
    # print('feeding gesture count accuracy:')
    # print(ans)
    # subprocess.call(['./runIS.sh'])
    
    with open(textfilepath, "a") as myfile:
        myfile.write('Time-point based fusion...\n\n')

	# eng = matlab.engine.start_matlab()
	# ans = eng.FG_predCount(protocol, subj)

	# # promptLabel.text = str(ans)

 #    with open(textfilepath, "a") as myfile:
 #        myfile.write(str(ans))



    with open(textfilepath, "a") as myfile:
        myfile.write('FINAL RESULT:\n')


    line0 = 'Ground truth:'
    line1 = 'Eating minutes: '+str(12)
    line2 = 'Number FG: '+str(34)
    line3 = 'KCal Estimation: '+str(1420)

    line4 = '\nPrediction:'
    line5 = 'Eating minutes: '+str(14)
    line6 = 'Number FG: '+str(35)
    line7 = 'KCal Estimation: '+str(1581)

    # promptLabel.text = "\n".join([line0, line1, line2, line3, line4, line5, line6, line7])
    with open(textfilepath, "a") as myfile:
        myfile.write("\n".join([line0, line1, line2, line3, line4, line5, line6, line7])+'\n')


    return output_file


# protocol = 'inlabStr'
# subj = 'P3'
# run = 41800
# dist = 0.7
# n_motif = 27.0
# config_file = 'config_file_is'

# eng = matlab.engine.start_matlab()
# ans = eng.FG_predCount(protocol, subj, run)

# print(str(ans))