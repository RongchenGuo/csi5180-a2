import argparse
import torch
import torchaudio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from transformers import BertTokenizer, BertModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
model = model.to(device)


def transform_label(label):
    if '(' in label and ')' in label:
        label = int(label[1])
        # (3, 2) (4, 1) (5, 0)
        if label > 2:
            return 1
        # (1, 4) (0, 5)
        if label < 2:
            return 0
        # (2, 3)
        return -1
    else:
        label = int(label)
        # 4 or 5
        if label > 3:
            return 1
        # 0 or 1 or 2
        if label < 3:
            return 0
        # 3
        return -1


def read_data(path='./dev.data'):
    data = []
    f = open(path, "r")
    for line in f:
        line = line.strip().split('\t')
        if len(line) == 7:
            data.append(line)
    f.close()
    data = pd.DataFrame(data=data, columns=['Topic_Id', 'Topic_Name', 'Sent_1', 'Sent_2', 'Label', 'Sent_1_tag', 'Sent_2_tag'])
    data['Label'] = data['Label'].map(lambda x: transform_label(x))
    # print(len(data))
    data.drop(index=data[data['Label'] == -1].index, inplace=True)
    data.reset_index(drop=True, inplace=True)
    # print(len(data))
    return data


def exact_match(sent1, sent2, alpha=0.):
    return 1 if sent1.strip() == sent2.strip() else 0


def edit_distance(sent1, sent2, alpha=0.81):
    sent1 = sent1.strip().split(" ")
    sent2 = sent2.strip().split(" ")
    wer = torchaudio.functional.edit_distance(sent1, sent2) / max(len(sent1), len(sent2))
    return 1 if wer < alpha else 0


def bert(sent1, sent2, alpha=5.0):
    tokenized_text = tokenizer([sent1, sent2], padding=True, return_tensors="pt").to(device)
    embedding = model(input_ids=tokenized_text['input_ids'])[1]
    return 1 if torch.dist(embedding[0], embedding[1], p=2) < alpha else 0


def tune_wer():
    alpha_list = np.arange(0.1, 1.0, 0.01)
    data = read_data(path='./train.data')
    x, y = [], []
    for a in tqdm(alpha_list):
        pred = [edit_distance(sent1=data['Sent_1'][i], sent2=data['Sent_2'][i], alpha=a) for i in range(len(data))]
        true = [data['Label'][i] for i in range(len(data))]
        accuracy = f1_score(true, pred, average='macro')
        x.append(a)
        y.append(accuracy)
    # plt.figure(figsize=(12, 8))
    plt.plot(x, y)
    plt.xlabel('alpha')
    plt.ylabel('F1 score')
    # plt.savefig('wer_alpha')
    plt.show()
    print("Best alpha = " + str(x[int(np.argmax(y).tolist())]))


def tune_bert():
    # alpha_list = np.arange(2., 3., 0.1)
    alpha_list = [1.0, 1.7, 2.0, 2.5, 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.]
    data = read_data(path='./train.data')
    x, y = [], []
    for a in alpha_list:
        # print("alpha = ", a) 
        pred = [bert(sent1=data['Sent_1'][i], sent2=data['Sent_2'][i], alpha=a) for i in tqdm(range(len(data)))]
        true = [data['Label'][i] for i in range(len(data))]
        accuracy = f1_score(true, pred, average='macro')
        x.append(a)
        y.append(accuracy)
        print("alpha = ", a, "\t", "f1 = ", accuracy)
    # plt.figure(figsize=(12, 8))
    plt.plot(x, y)
    plt.xlabel('alpha')
    plt.ylabel('F1 score')
    plt.savefig('bert_alpha')
    # plt.show()
    print("Best alpha = " + str(x[int(np.argmax(y).tolist())]))


def main():
    parser = argparse.ArgumentParser(description='paraphrase.py')
    parser.add_argument('--mode', type=str, default='test',
                        help='choose from [train, dev, test]')
    parser.add_argument('--alg', type=str, default='wer',
                        help='choose from [em, wer, bert]')
    args = parser.parse_args()
    print(args)
    data = read_data(path='./' + args.mode + '.data')
    if args.alg == 'em':
        algorithm = 'Exact Match'
        method = exact_match
    elif args.alg == 'wer':
        algorithm = 'Edit Distance'
        method = edit_distance
    else:
        algorithm = 'BERT'
        method = bert

    pred = [method(sent1=data['Sent_1'][i], sent2=data['Sent_2'][i]) for i in range(len(data))]
    true = [data['Label'][i] for i in range(len(data))]
    print("Mode = " + args.mode + "\tAlgorithm = " + algorithm)
    print(classification_report(true, pred, target_names=['non-paraphrase', 'paraphrase']))

    # print("Mode = " + args.mode + "\tAlgorithm = " + algorithm)
    cm = confusion_matrix(true, pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.savefig(args.alg)
    plt.show()


# tune_wer()
# tune_bert()
main()
