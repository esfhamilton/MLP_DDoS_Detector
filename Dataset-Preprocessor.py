'''
@Title: Dataset Preprocessor
@Created: 06/02/2021
@Last Modified: 25/03/2021
@Author: Ethan Hamilton

Used for reformatting and preprocessing network traffic flow 
datasets prior to training within an ANN. Contains functions for 
trimming, decomposition, feature selection, NaN removal, encoding and scaling.
'''

import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import sklearn
from sklearn.preprocessing import MinMaxScaler


# Trims CSV files with more than 1048575 rows  
def trim_csv():
    # CSV file names in target folder
    csvs = os.listdir('Datasets/CIC-DDoS2019/03-11')

    # Remove none csv files
    csvs = [csv for csv in csvs if '.csv' in csv]
    
    # Amount of rows for each partition (1048575 is the maximum that Excel handles) 
    chunksize = 1048575

    # Imports 'chunksize' rows of data from each file and creates a smaller CSV file with them 
    for csv in csvs:
        for chunk in pd.read_csv('Datasets/CIC-DDoS2019/01-12/{}'.format(csv),dtype='unicode',chunksize=chunksize):
            print('Creating Trimmed {}'.format(csv))
            chunk.to_csv('Datasets/CIC-DDoS2019/01-12/Trimmed {}'.format(csv), index=False)
            print('Trimmed {} has been created successfully'.format(csv))
            break 

# Separates each class into new individual csv files
def class_separator():
    csvs = os.listdir('Datasets/CSE-CIC-IDS2018')

    # Remove none csv files
    csvs = [csv for csv in csvs if '.csv' in csv]
    
    for csvIndex, csv in enumerate(csvs):        
        df = pd.read_csv('Datasets/CSE-CIC-IDS2018/{}'.format(csv),sep='\s*,\s*',engine='python',dtype='unicode')
        print("Loaded sample data")
        labels = []
        newDatasets = {}

        for i, row in enumerate(df.iterrows()):
            if(i%10000==0 and i>0):
                print(str(i)+' rows processed')
                
            # Add data from row to corresponding list
            tempList = []
            for i in range(len(row[1])):
                tempList.append(row[1][i])
                
            # Initialise a new list for each distinct label     
            if(row[1]['Label'] not in labels):
                labels.append(row[1]['Label'])
                newDatasets[row[1]['Label']] = []
                
            newDatasets[row[1]['Label']].append(tempList)

        # Create new directory to store decomposed class files
        os.mkdir('Datasets/CSE-CIC-IDS2018/{}'.format(csvs[csvIndex].strip('.csv') + ' Decomposition'))
        for label in labels:
            newDataset = DataFrame(newDatasets[label], columns=df.columns)
            # Replace Test1 with Name of original dataset + Decomposition
            newDataset.to_csv('Datasets/CSE-CIC-IDS2018/{}/{}.csv'.format(csvs[csvIndex].strip('.csv')+ ' Decomposition',label), index=False)



# Check for highly correlated features				
def corFeatures(df):
	cor = df.corr()
	for f1 in df.columns:
		for f2 in df.columns:
			if(abs(cor[f1][f2]) > 0.95 and f1 != f2): 
				print("Features: {} and {}".format(f1,f2))
				print("Correlation:{}".format(cor[f1][f2]))		

# Preprocessing for the main datasets 
# For leaving attack out, temporarily remove attack topFolders directory
def main():
	# Placeholder for preprocessed dataset
	newDF = DataFrame()

	# Loop through every Decomposition folder 
	topFolders = os.listdir('Datasets/CSE-CIC-IDS2018')

	# Limits the amount 
	chunksize = 117514
	for folder1 in topFolders:
		DFolders = os.listdir('Datasets/CSE-CIC-IDS2018/{}'.format(folder1))
		for folder2 in DFolders:
			DFiles = os.listdir('Datasets/CSE-CIC-IDS2018/{}/{}'.format(folder1,folder2))
			for csv in DFiles:
				for chunk in pd.read_csv('Datasets/CSE-CIC-IDS2018/{}/{}/{}'.format(folder1,folder2,csv),
                                                         sep='\s*,\s*',engine='python',chunksize=chunksize):

					# Remove both socket and highly correlated features
					dropCols = ['Timestamp','Unnamed: 0','Flow ID','Src IP','Src Port',
                                                    'Dst IP','Dst Port','Source IP','Source Port','Destination IP',
                                                    'Destination Port','SimillarHTTP','Bwd Packet Length Mean',
                                                    'Bwd Pkt Len Mean','Fwd Packet Length Min','Fwd Pkt Len Min',
                                                    'Min Packet Length','Pkt Len Min','Idle Mean','Subflow Fwd Bytes',
                                                    'Subflow Fwd Byts','Subflow Bwd Bytes','Subflow Bwd Byts',
                                                    'Subflow Fwd Packets','Subflow Bwd Packets','Subflow Fwd Pkts',
                                                    'Subflow Bwd Pkts','Fwd Header Length.1','RST Flag Count',
                                                    'RST Flag Cnt','Fwd IAT Max','Flow IAT Max','Flow IAT Min',
                                                    'Avg Fwd Segment Size','Fwd IAT Total','Fwd IAT Tot',
                                                    'Packet Length Mean','Pkt Len Mean','Fwd Packets/s',
                                                    'Fwd Pkts/s','Flow IAT Std','Average Packet Size','Pkt Size Avg',
                                                    'Total Length of Bwd Packets','TotLen Bwd Pkts','Flow IAT Mean']
					df = chunk.drop(columns=list(chunk.columns.intersection(dropCols)))
					

					# Remove Infinite and NaN values
					df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

					# Logging
					print(folder1,folder2,csv)

					# Encode 'Label' column and normalize data
					if (df['Label'][0].upper() == 'BENIGN'): 
						df['Label'] = 0  
						newDF = pd.concat([newDF,df.head(383648)])
						#newDF = pd.concat([newDF,df.head(104857)])
					else: 
						df['Label'] = 1 
						newDF = pd.concat([newDF,df])

					break
	
	# Scale the Data
	scaler = MinMaxScaler()
	newDF = DataFrame(scaler.fit_transform(newDF), columns=df.columns)
	# Ensures an equal number of benign and malicious traffic
	print("Number of attacks:")
	print(len(newDF[newDF['Label']==1]))
	print("Number of benign:")
	print(len(newDF[newDF['Label']==0]))
	# Write preprocessed dataset to new csv
	newDF.to_csv('Datasets/2018_Preprocessed.csv', index=False)	


# Preprocessing for the datasets with 1 attack left out
def leaveOneOut():
	
	# STEP 1: Preprocess training dataset 	

	# Placeholders for preprocessed dataset
	benignDF = DataFrame()	
	attackDF = DataFrame()
	benignTestDF = DataFrame()	
	attackTestDF = DataFrame()
	trainDF = DataFrame()

	# Loop through every Decomposition folder 
	topFolders = os.listdir('Datasets/CIC-DDoS2019')

	# Limits number of rows read at a time
	chunksize = 150000
	for folder1 in topFolders:
		DFolders = os.listdir('Datasets/CIC-DDoS2019/{}'.format(folder1))

		for folder2 in DFolders:
			DFiles = os.listdir('Datasets/CIC-DDoS2019/{}/{}'.format(folder1,folder2))
			for csv in DFiles:
				for chunk in pd.read_csv('Datasets/CIC-DDoS2019/{}/{}/{}'.format(folder1,folder2,csv),sep='\s*,\s*',engine='python',chunksize=chunksize):

                                        # Remove both socket and highly correlated features
                                        dropCols = ['Timestamp','Unnamed: 0','Flow ID','Src IP','Src Port','Dst IP','Dst Port','Source IP','Source Port','Destination IP','Destination Port','SimillarHTTP','Bwd Packet Length Mean','Bwd Pkt Len Mean','Fwd Packet Length Min','Fwd Pkt Len Min','Min Packet Length','Pkt Len Min','Idle Mean','Subflow Fwd Bytes','Subflow Fwd Byts','Subflow Bwd Bytes','Subflow Bwd Byts','Subflow Fwd Packets','Subflow Bwd Packets','Subflow Fwd Pkts','Subflow Bwd Pkts','Fwd Header Length.1','RST Flag Count','RST Flag Cnt','Fwd IAT Max','Flow IAT Max','Flow IAT Min','Avg Fwd Segment Size','Fwd IAT Total','Fwd IAT Tot','Packet Length Mean','Pkt Len Mean','Fwd Packets/s','Fwd Pkts/s','Flow IAT Std','Average Packet Size','Pkt Size Avg', 'Total Length of Bwd Packets','TotLen Bwd Pkts','Flow IAT Mean']
                                        df = chunk.drop(columns=list(chunk.columns.intersection(dropCols)))
					

                                        # Remove Infinite and NaN values
                                        df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

                                        # Logging
                                        print(folder1,folder2,csv)
                                        print(df.shape)
                                        if ((folder2 == "DrDoS_DNS Decomposition" and csv == "BENIGN.csv") or csv == "WebDDoS.csv"):
                                            if (csv == "BENIGN.csv"):
                                                df['Label'] = 0
                                                benignTestDF = pd.concat([benignTestDF,df.iloc[:50000]])
                                            else:
                                                df['Label'] = 1
                                                attackTestDF = pd.concat([attackTestDF,df.iloc[:50000]])
                                        else:
                                            if (df['Label'][0].upper() == 'BENIGN'): 
                                                    df['Label'] = 0  
                                                    benignDF = pd.concat([benignDF,df])
                                            else: 
                                                    df['Label'] = 1 
                                                    attackDF = pd.concat([attackDF,df])

                                        break	
	

	if (len(benignDF) > len(attackDF)):
		benignDF = benignDF.iloc[:len(attackDF)]
	else:
		attackDF = attackDF.iloc[:len(benignDF)]

	benignTestDF = benignTestDF.iloc[:len(attackTestDF)]
	trainDF = pd.concat([trainDF,benignDF])
	trainDF = pd.concat([trainDF,attackDF])
	trainDF = pd.concat([trainDF,benignTestDF])
	trainDF = pd.concat([trainDF,attackTestDF])
	
	# Scale the Data
	scaler = MinMaxScaler()
	trainDF = DataFrame(scaler.fit_transform(trainDF), columns=df.columns)

	testDF = trainDF.tail(len(attackTestDF)+len(benignTestDF)).sample(frac=1)
	trainDF = trainDF.head(len(trainDF)-(len(attackTestDF)+len(benignTestDF))).sample(frac=1)

	trainDF.to_csv('Datasets/train.csv', index=False)
	testDF.to_csv('Datasets/test.csv', index=False)
	

leaveOneOut()
