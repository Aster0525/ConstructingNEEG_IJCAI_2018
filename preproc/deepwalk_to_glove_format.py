if __name__ == '__main__':
    emb_file = 'data/deepwalk_128_unweighted_with_args.txt'
    vocab_file = 'data/encoding_with_args.csv'
    output_file = 'temp/deepwalk.txt'

    id2word = {}
    with open(vocab_file, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            id2word[int(line[0])] = line[1]

    lines = open(emb_file, 'r').readlines()[1:]
    print(len(lines))
    lines = [line.strip().split(' ') for line in lines]
    lines = sorted(lines, key=lambda line: int(line[0]))

    output_file = open(output_file, 'w')
    for line in lines:
        word_id = int(line[0])
        word_text = id2word[word_id]
        output_file.write(' '.join([word_text] + line[1:]) + '\n')
    output_file.close()
