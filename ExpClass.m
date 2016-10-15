%Demo of HSI data loading and classification
clear;close all;clc
addpath(genpath(pwd))
%%
dataInfo=HsiDataLoad('PaviaUniv');
dataInfo.splitInfo=SplitData(dataInfo.gt_data,'fix_num',100,true);

train_feat=VectorIndexing3D(dataInfo.spectral_data,dataInfo.splitInfo.train_idx);
test_feat=VectorIndexing3D(dataInfo.spectral_data,dataInfo.splitInfo.test_idx);
train_label=dataInfo.splitInfo.train_label;
test_label=dataInfo.splitInfo.test_label;
%%
[w,val_loss] = MLRTrainAL(train_feat'/1000,train_label', 0.1,0.0001,100);
[pred_label,pred_prob]=MLREval(test_feat/1000,dataInfo.num_class, w);

%
[clsStat,mat_conf]=GetAccuracy(test_label,pred_label);
disp(['Overall Accuracy:',num2str(clsStat.OA),', Kappa Coeffcient:', num2str(clsStat.Kappa)]);
color_map=GetColorMap(16);
mat_pred_label=zeros(size(dataInfo.gt_data));
mat_pred_label(dataInfo.splitInfo.test_idx)=pred_label;
class_map=GetClassMap(mat_pred_label,color_map);
figure(1);set(gcf,'position',[500 200 400 600])
imagesc(class_map);

