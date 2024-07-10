def cola_label_map(dataset, label_count):
    texts = []
    labels = []

    for sent in dataset:
        # print(sent['label'])
        texts.append(sent['sentence'])
        labels.append(label_count+1 if sent['label'] is 1 else label_count)

    return texts, labels


def sst2_label_map(dataset, label_count):
    texts = []
    labels = []
    # for data_set in ["pos", "neg"]:
    #     for text_file in (split_dir/label_dir).iterdir():
    #         texts.append(text_file.read_text())
    #         labels.append(0 if label_dir is "neg" else 1)

    for sent in dataset:
        # print(sent['label'])
        texts.append(sent['sentence'])
        labels.append(label_count+1 if sent['label'] is 1 else label_count)

    return texts, labels