#coding:utf8
# Run this code to train our SGNN model.
# Generally we can train a model in about 1400 seconds (the code will automatically terminate by using early stop) using one Tesla P100 GPU.
from gnn_with_args import *
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='../data/corpus_index_train0_with_args_all_chain.data')
    parser.add_argument('--dev_data', type=str, default='../data/corpus_index_dev_with_args_all_chain.data')
    parser.add_argument('--test_data', type=str, default='../data/corpus_index_test_with_args_all_chain.data')
    parser.add_argument('--dev_index', type=str, default='../data/dev_index.pickle')
    parser.add_argument('--test_index', type=str, default='../data/test_index.pickle')
    parser.add_argument('--vocab_file', type=str, default='../data/encoding_with_args.csv')
    parser.add_argument('--emb_file', type=str, default='../data/deepwalk_128_unweighted_with_args.txt')
    parser.add_argument('--pad_idx', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=19950125)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--l2_penalty', type=float, default=0.00001)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--t', type=int, default=2)
    parser.add_argument('--margin', type=float, default=0.015)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=520)
    parser.add_argument('--patients', type=int, default=500)
    parser.add_argument('--em_r', type=int, default=10)
    parser.add_argument('--metric', type=str, default='euclid')
    parser.add_argument('--event_repr', type=str, default='ntn')
    parser.add_argument('--pretrained_event_model', type=str, default='')
    parser.add_argument('--save_prefix', type=str, default='../models/model')
    option = parser.parse_args()

    assert option.event_repr in ['cat', 'ntn', 'role_factor', 'low_rank_ntn']

    torch.manual_seed(option.random_seed)

    dev_data=Data_data(pickle.load(open(option.dev_data,'rb')))
    test_data=Data_data(pickle.load(open(option.test_data,'rb')))
    train_data=Data_data(pickle.load(open(option.train_data, 'rb')))
    # ans=pickle.load(open('../data/dev.answer','rb'))
    ans = None
    dev_index=pickle.load(open(option.dev_index,'rb'))
    test_index=pickle.load(open(option.test_index,'rb'))
    print('train data prepare done')
    word_id,id_vec,word_vec=get_hash_for_word(option.emb_file, option.vocab_file, option.pad_idx)
    print('word vector prepare done')

    # if len(sys.argv)==9:
    #     L2_penalty,MARGIN,LR,T,BATCH_SIZE,EPOCHES,PATIENTS,METRIC=sys.argv[1:]
    # else:
    HIDDEN_DIM = option.hidden_dim
    L2_penalty = option.l2_penalty
    LR = option.lr
    T = option.t
    MARGIN = option.margin
    BATCH_SIZE = option.batch_size
    EPOCHES = option.epochs
    PATIENTS = option.patients
    METRIC = option.metric

        # if METRIC=='euclid':  #   
        #     L2_penalty=0.00001
        #     LR=0.0001
        #     BATCH_SIZE=1000
        #     MARGIN=0.015
        #     PATIENTS=500
        # if METRIC=='dot':  # 
        #     # LR=0.004
        #     MARGIN=0.5
        # if METRIC=='cosine': # 
        #     # LR=0.001
        #     MARGIN=0.05
        # if METRIC=='norm_euclid': # 
        #     # LR=0.0011
        #     MARGIN=0.07
        # if METRIC=='manhattan': # 
        #     # LR=0.0015
        #     MARGIN=4.5
        # if METRIC=='multi': # 
        #     # LR=0.001
        #     MARGIN=0.015
        # if METRIC=='nonlinear': # 
        #     # LR=0.001
        #     MARGIN=0.015
    start=time.time()
    best_acc,best_epoch=train(dev_index,test_index,word_vec,ans,train_data,dev_data,test_data,float(L2_penalty),float(MARGIN),float(LR),int(T),int(BATCH_SIZE),int(EPOCHES),int(PATIENTS),int(HIDDEN_DIM),option.em_r,METRIC, option.event_repr, option.pretrained_event_model, option.save_prefix)
    end=time.time()
    print ("Run time: %f s" % (end-start))
    with open('../output/best_result.txt','a') as f:
        f.write('Best Acc: %f, Epoch %d , L2_penalty=%s ,MARGIN=%s ,LR=%s ,T=%s ,BATCH_SIZE=%s ,EPOCHES=%s ,PATIENTS=%s, HIDDEN_DIM=%s, METRIC=%s\n' % (best_acc,best_epoch,L2_penalty,MARGIN,LR,T,BATCH_SIZE,EPOCHES,PATIENTS,HIDDEN_DIM,METRIC))
    f.close()


if __name__ == '__main__':
    main()

# 事件表示：事件链条的多维分布表示，加入频率和共现频次信息
# 构建Graph: 统计bigram-过滤低频,删除自环,高频事件处理-图构建-计算概率
# Context Extension By Ranking
# Highway Networks
# SRU
# Attention
# Subgraph Embedding
# Adam
