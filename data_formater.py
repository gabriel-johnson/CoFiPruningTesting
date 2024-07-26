def cola_label_map(dataset, label_count):
    texts = []
    labels = []

    for sent in dataset:
        # print(sent['label'])
        texts.append("<cola>"+sent['sentence'])
        labels.append(label_count+1 if sent['label'] is 1 else label_count)

    return texts, labels


def sst2_label_map(dataset, label_count):
    texts = []
    labels = []
   
    for sent in dataset:
        # print(sent['label'])
        texts.append("<sst2>"+sent['sentence'])
        labels.append(label_count+1 if sent['label'] is 1 else label_count)
    

    return texts, labels

def map_labels(task, dataset, label_count):
    if task == "cola":
        return cola_label_map(dataset, label_count)
    
    elif task == "sst2":
        return sst2_label_map(dataset, label_count)
    
    