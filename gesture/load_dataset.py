import torch
from torch.utils.data import Dataset
import csv
import json

class gesture_dataset(Dataset):
    def __init__(self,csv_file):
        #stores each row of features (61 landmark values) (21 landmarks x 3 (x,y,z) = 61)
        self.samples=[]
        #store only labels 
        self.labels=[]
        #convert class name ('fist') into integers ('2')
        self.label_map={}
        #for converting prediction results back into original string labels
        self.reverse_label_map={}
        label_index=0

        #open my csv file as read only
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            #skip the headers
            next(reader)
            #each row looks like: ['fist', 0.2, 0.4, ...]
            for row in reader:
                label=row[0]
                #convert strings to floats
                features=list(map(float,row[1:]))

                #if label not in label_map yet,assign it with a new integer index
                if label not in self.label_map:
                    #label map will look like this: {'fist':0, 'peace':1, ...}
                    self.label_map[label]=label_index
                    self.reverse_label_map[label_index]=label
                    label_index+=1

                self.samples.append(features)
                self.labels.append(self.label_map[label])
            
            #writing label map into json file
            with open("gesture_label_map.json","w") as f:
                json.dump(self.label_map,f,indent=4)
        
        #convert list to tensors 
        self.samples=torch.tensor(self.samples, dtype=torch.float32)
        #here, torch.long == torch.int64, just a naming convention in pytorch
        self.labels=torch.tensor(self.labels, dtype=torch.long)
    
    #len and getitem must be implemented!!!
    #returns the size of dataset
    def __len__(self):
        return len(self.samples)
    #returns a single sample (input + label)
    def __getitem__(self, index):
        return self.samples[index],self.labels[index]
    