%Code for Random Forest
%Rob Romijnders
% 22 April 2016

clear all
close all
clc

X_train = csvread('train_data.csv',1,1);
X_val = csvread('assembling_data.csv',1,1);
X_test = csvread('test_data.csv',1,1);

y_train = X_train(:,1);
X_train = X_train(:,2:end);
y_val = X_val(:,1);
X_val = X_val(:,2:end);
y_test = X_test(:,1);
X_test = X_test(:,2:end);

RF = Stochastic_Bosque(X_train,y_train);




%%

disp('Start bagging')
tic
B = TreeBagger(80,X_train,y_train,'OOBPrediction','On',...
    'Method','classification');
time = toc;
disp(sprintf('Finished bagging in %.3f seconds',time))

%Plot OOB
OOB = oobError(B);
plot(OOB)
xlabel('Number of trees');
ylabel('OOB classification error');

%Evaluate metrics
y_val_pred = str2num(cell2mat(predict(B,X_val)));
acc_val  = mean(y_val == y_val_pred);
disp(sprintf('Validation accuracy is %.4f',acc_val))


