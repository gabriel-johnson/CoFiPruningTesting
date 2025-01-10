# def cola_label_map(dataset, label_count):
#     texts = []
#     labels = []

#     for sent in dataset:
#         # print(sent['label'])
#         texts.append("<cola>"+sent['sentence'])
#         labels.append(label_count+1 if sent['label'] is 1 else label_count)

#     return texts, labels


# def sst2_label_map(dataset, label_count):
#     texts = []
#     labels = []
   
#     for sent in dataset:
#         # print(sent['label'])
#         texts.append("<sst2>"+sent['sentence'])
#         labels.append(label_count+1 if sent['label'] is 1 else label_count)
    

    # return texts, labels

def map_labels(task, dataset, label_count):
   
    def append_token(example):
       
        if task == "qqp":

            example['question1'] = "<" + task + "> " + example["question1"]
            example['question2'] = "<" + task + "> " + example["question2"]
    
        elif task == "qnli":
            example['question'] = "<" + task + "> " + example["question"]
            example['sentence'] = "<" + task + "> " + example["sentence"]

        
        elif task == "sst2":
            example["sentence"] = "<" + task + "> " + example["sentence"]

        elif task == "mrpc":

            example['sentence1'] = "<" + task + "> " + example["sentence1"]
            example['sentence2'] = "<" + task + "> " + example["sentence2"]
    
        return example



    return dataset.map(append_token)

    
    