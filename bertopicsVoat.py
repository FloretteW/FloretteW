#import packages

from asyncio.windows_events import NULL
import csv

import pandas as pd 
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

#data = open("GreatAwakening_submissions.ndjson")


start1 = "\"title\": "
idx_start1 = 1
end1 = "\"user\": "
idx_end1 = 1
start2 = "\"body\": "
idx_start2 = 1
end2= "}"
idx_end2 = 1
raw_chara=r"\a"
raw_charb=r"\b"
raw_charn=r"\n"
raw_chars=r"\s"
raw_chart=r"\t"
raw_chare=r"\e"
raw_charu=r"\u"
raw_charr=r"\r"
raw_charv=r"\v"


# with open(r"C:/Users/flore/Voat_dataset_QAnon/GreatAwakening_submissions.ndjson", "r") as readFile:
#     with open(r"C:/Users/flore/Voat_dataset_QAnon/GreatAwakening_submissions_bodyonly.ndjson", "w+") as writeFile:
#         for str in readFile:
#             idx_start1 = str.index(start1) + len(start1)
#             idx_end1 = str.index(end1) -2
#             towrite = str[idx_start1:idx_end1]
#             idx_start2 = str.index(start2) + len(start2)
#             idx_end2 = str.index(end2) -2
#             towrite = str[idx_start2:idx_end2]
#             writeFile.write(towrite)
#         for str in writeFile:    
#             mapping = [ (raw_chara, ' '), (raw_charb, ' '), (raw_charn, ' '), (raw_chars, ' '), (raw_chart, ' '), (raw_chare, ' '), (raw_charu, ' '), (raw_charr, ' '), (raw_charv, ' '), ('\\', " ") ]
#             for k, v in mapping:
#                  str = str.translate('\\', " ")
#     writeFile.close()
# readFile.close()
# with open(r"C:/Users/flore/Voat_dataset_QAnon/GreatAwakening_submissions_bodyonly.ndjson", "w+") as writeFile: 
#         for str in writeFile:    
#             mapping = [ (raw_chara, ''), (raw_charb, ''), (raw_charn, ''), (raw_chars, ''), (raw_chart, ''), (raw_chare, ''), (raw_charu, ''), (raw_charr, ''), (raw_charv, '') ]
#             for k, v in mapping:
#                 str = str.replace(k, v)
# writeFile.close()
#df = pd.read_csv("C:/Users/flore/Voat_dataset_QAnon/GreatAwakening_submissions_bodyonly.csv")
df = pd.read_csv('C:/Users/flore/Voat_dataset_QAnon/GreatAwakening_comments_bodyonly.csv', delimiter=',', sep=NULL)
df = df[0:60]
#docs=df.to_list()
model = BERTopic(verbose=True)
 
# with open('C:/Users/flore/Voat_dataset_QAnon/GreatAwakening_submissions_bodyonly.csv', 'r') as read_obj: # read csv file as a list of lists
#     csv_reader = csv.reader(read_obj) # pass the file object to reader() to get the reader object
#     docs = str(list(csv_reader)) # Pass reader object to list() to get a list of lists
#convert to list 
#docs = df.text_to_list()
#print(docs)
mylist=list(df)
topics, probabilities = model.fit_transform(mylist)