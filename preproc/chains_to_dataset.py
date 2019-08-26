import torch
import pickle
import random
from collections import Counter


def generate_samples(chain, events, window_size=5):
    samples = []
    for i in range(len(chain) - 1):
        e1 = chain[i]
        window = chain[i+1 : i+window_size]
        for e2 in window:
            e_neg = random.choice(events) + []
            protagonist = e_neg[-1]
            for i in range(4):
                if e_neg[i] == protagonist:
                    e_neg[i] = e1[-1]
            samples.append((e1, e2, e_neg))
    return samples


if __name__ == '__main__':
    input_file = 'data/corpus_index_train0_with_args_all_chain.data'
    output_file = 'temp/train_deepwalk.txt'
    vocab_file = 'data/encoding_with_args.csv'

    random.seed(19950125)

    id2word = {}
    with open(vocab_file, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            id2word[int(line[0])] = line[1]

    a = pickle.load(open(input_file, 'rb'))
    a1 = a[1]
    a2 = a[2]

    verb = a1[:, 0:13]
    subj = a1[:, 13:26]
    obj = a1[:, 26:39]
    iobj = a1[:, 39:52]

    context_verb = verb[:, :8]
    context_subj = subj[:, :8]
    context_obj = obj[:, :8]
    context_iobj = iobj[:, :8]

    candidate_verb = verb[:, 8:]
    candidate_subj = subj[:, 8:]
    candidate_obj = obj[:, 8:]
    candidate_iobj = iobj[:, 8:]

    correct_verb = []
    correct_subj = []
    correct_obj = []
    correct_iobj = []
    for i in range(a1.size(0)):
        idx = a2[i].item()
        correct_verb.append(candidate_verb[i][idx])
        correct_subj.append(candidate_subj[i][idx])
        correct_obj.append(candidate_obj[i][idx])
        correct_iobj.append(candidate_iobj[i][idx])

    correct_verb = torch.stack(correct_verb).unsqueeze(1)
    correct_subj = torch.stack(correct_subj).unsqueeze(1)
    correct_obj = torch.stack(correct_obj).unsqueeze(1)
    correct_iobj = torch.stack(correct_iobj).unsqueeze(1)

    context_verb = torch.cat((context_verb, correct_verb), 1)
    context_subj = torch.cat((context_subj, correct_subj), 1)
    context_obj = torch.cat((context_obj, correct_obj), 1)
    context_iobj = torch.cat((context_iobj, correct_iobj), 1)

    events = []
    chains = []
    for i in range(a1.size(0)):
        chain = []
        counter = Counter()
        for j in range(9):
            verb = context_verb[i][j].item()
            subj = context_subj[i][j].item()
            obj = context_obj[i][j].item()
            iobj = context_iobj[i][j].item()
            verb = id2word[verb]
            subj = id2word[subj]
            obj = id2word[obj]
            iobj = id2word[iobj]
            event = [subj, verb, obj, iobj]
            counter.update([subj])
            events.append(event)
            chain.append(event)
        protagonist = counter.most_common(1)[0][0]
        for event in chain:
            event.append(protagonist)
        chains.append(chain)

    samples = []
    for chain in chains:
        samples += generate_samples(chain, events)
    random.shuffle(samples)
    output_file = open(output_file, 'w')
    for e1, e2, e_neg in samples:
        e1 = '|'.join(e1[:-2])
        e2 = '|'.join(e2[:-2])
        e_neg = '|'.join(e_neg[:-2])
        s = ', '.join([e1, e2, e_neg])
        output_file.write(s + '\n')
    output_file.close()
