import h_model 
import os, argparse, sys, time
import numpy as np
import utils
import torch
import torch.nn as nn
import torch.optim as optim

def save(model, save_filename):
    with open(save_filename, "wb") as f:
        torch.save(model, f)
    print('Saved as %s' % save_filename)


def output_s(message, save_filename):
    print (message)
    with open(save_filename, 'a') as out:
        out.write(message + '\n')


def vectorize_data(filename, vocab, id2vocab):
    # words are considered in a document level
    words = open(filename).read().replace('\n', '<eos>').strip().split()
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            # put one hot vector: len(vocab) as a index
            vocab[word] = len(vocab) 
            id2vocab[vocab[word]] = word
            # present input data as a sequence of  one-hot vector
        dataset[i] = vocab[word]
    return dataset, vocab, id2vocab

def make_input_data(noise_data, data, seq_len, alph, noise_type, vocab): # training, dev, or test
    max_char_num = 20
    if 'INS' in noise_type:
        max_char_num += 1
    X_vec = np.zeros((int(len(noise_data)/seq_len), seq_len, max_char_num), dtype=np.int32)
    mask_vec = np.zeros((int(len(noise_data)/seq_len), seq_len, max_char_num, max_char_num), dtype=np.int32)
    Y_vec = np.zeros((int(len(data)/seq_len), seq_len, 1), dtype=np.int32)
    # easy minibatch
    # https://docs.python.org/2.7/library/functions.html?highlight=zip#zip
    for m, mini_batch_tokens in enumerate(zip(*[iter(noise_data)]*seq_len)):
        mask_mini_batch = np.zeros((seq_len, max_char_num, max_char_num), dtype=np.int32)
        x_mini_batch = np.zeros((seq_len, max_char_num), dtype=np.int32)
        y_mini_batch = np.zeros((seq_len, 1), dtype=np.int32)
        for j, token in enumerate(mini_batch_tokens):
            bin_all, mask_all = utils.vec_char(token, alph, max_char_num)
            x_mini_batch[j] = bin_all
            for k in range(max_char_num):
                mask_mini_batch[j][k] = mask_all
            true_token = data[m*seq_len:(m+1)*seq_len][j]
            y_mini_batch[j] = vocab[true_token]

        X_vec[m] = x_mini_batch
        mask_vec[m] = mask_mini_batch
        Y_vec[m] = y_mini_batch

        percentage = int(m*100. / (len(data)/seq_len))
        sys.stdout.write("\r%d %%" % (percentage))
        #print(str(percentage) + '%'),
        sys.stdout.flush()
    print('\n', X_vec.shape, mask_vec.shape, Y_vec.shape)
    return torch.LongTensor(X_vec), torch.LongTensor(mask_vec), torch.LongTensor(Y_vec)


def evaluate(X_valid, mask_valid, Y_valid, batch_size, seq_len, ntokens, args):
    model.eval()
    correct = 0
    total = 0
    #hidden = model.init_hidden(batch_size)
    with torch.no_grad():
        for batch, i in enumerate(range(0, X_valid.size(0) - 1, batch_size)):
            X, mask, Y = utils.get_batch(X_valid, mask_valid, Y_valid, batch_size, i)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            if args.num == 1:
                output, hidden = model(X, mask)
            if args.num in [2,3]:
                output, hidden, seq_output = model(X, mask)
            output = output.view(-1, ntokens)
            _, predicted = torch.max(output.data, 1)
            #print (Y.shape)
            Y = Y.view(-1)
            #print (predicted.shape, Y.shape)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()
        return 100*correct/total
        # check output
def decode_word(X,  id2vocab):
    return ' '.join(id2vocab[x.item()] for x in X)

def check(X_valid, mask_valid, Y_valid, valid_noise_tokens, valid_tokens, id2vocab, ntokens, seq_len, args):
    """
    X_valid: seq_len, seq_len, d_input
    Y_valid: seq_len, seq_len, 1
    """
    # for i in range(10):
    #     print ("="*10 + id2vocab[i])
    n = 2
    srcs = list(zip(*[iter(valid_noise_tokens)]*seq_len))[:n]

    for j in range(n):
        src_j = " ".join(srcs[j])
        x_raw, mask_raw, y_raw = X_valid[np.array([j])], mask_valid[np.array([j])], Y_valid[np.array([j])] # "np.array" to make the dim 3
        ref_j = decode_word(y_raw.view(-1), id2vocab)
        if args.num == 1:
            output, hidden = model(x_raw, mask_raw)
        if args.num in [2, 3]:
            output, hidden, seq_output = model(x_raw, mask_raw)
        output = output.view(-1, ntokens)
        _, predicted = torch.max(output.data, 1)
        print (predicted)
        pred_j = decode_word(predicted, id2vocab)

        print('example #', str(j+1))        
        print('SRC: ', src_j)
        print('REF: ', ref_j)        
        print('PRD: ', pred_j)
    return


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', default=100, type=int,
        help='number of epochs to learn')
    parser.add_argument('--d_hidden', '-d', default=650, type=int,
        help='number of units in hidden layers')
    parser.add_argument('--seq_len', '-b', type=int, default=20,
        help='learning minibatch size')

    parser.add_argument('--source', default="PER")
    parser.add_argument('--target', default="W_PER")

    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--renew_data', action='store_true')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--d_emb', type=int, default=512)
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--h', type=int, default=8)
    parser.add_argument('--num', type=int, default=3)
    parser.add_argument('--beta', type=float, default=1.0)


    device = torch.device("cuda")
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    source = args.source
    target = args.target
    lr = args.lr


    base_path = os.path.dirname(os.path.realpath(__file__))
    text_data_dir = os.path.join(base_path,'./data/')
    data_dir = os.path.join(base_path,'data/')
    output_dir = os.path.join(base_path,'output/')


    ######################################################
    ###########      DATA PREPARE              ###########
    ######################################################

    vocab, id2vocab = {}, {}
    train_vec, vocab, id2vocab = vectorize_data(text_data_dir + 'ptb.train.txt', vocab, id2vocab)
    

    valid_vec, vocab, id2vocab = vectorize_data(text_data_dir + 'ptb.valid.txt', vocab, id2vocab)
    
    test_vec, vocab, id2vocab = vectorize_data(text_data_dir + 'ptb.test.txt', vocab, id2vocab)
    test_noise_filename = text_data_dir + 'ptb.test.' + target + '.txt'
    test_noise_tokens = open(test_noise_filename).read().strip().split()
    test_filename = text_data_dir + 'ptb.test.txt'
    test_tokens = open(test_filename).read().replace('\n', '<eos>').strip().split()
    


    alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:;'*!?`$%&(){}[]-/\@_#" 

    
    
    X_test = torch.load(data_dir + 'X_test_{}.pt'.format(target))
    mask_test = torch.load(data_dir + 'mask_test_{}.pt'.format(target))
    Y_test = torch.load(data_dir + 'Y_test_{}.pt'.format(target))

    
    X_test, mask_test, Y_test = X_test.to(device), mask_test.to(device), Y_test.to(device)
    ######################################################
    ###########    MODEL AND TRAINING CONFIG   ###########
    ######################################################
    model_name = "{}_{}_beta_{}_emb_{}_h_{}_hidden_{}_n_{}_lr_{}_bs_{}_check_10".format(args.num, source, args.beta, args.d_emb, args.h, args.d_hidden, args.n, args.lr, args.batch_size)
    #model_name = "{}_{}_beta_{}_emb_{}_h_{}_hidden_{}_n_{}_lr_{}_bs_{}".format(args.num, source, args.beta, args.d_emb, args.h, args.d_hidden, args.n, args.lr, args.batch_size)

    global message_filename 
    message_filename = output_dir + 'cross_{}_{}_'.format(source, target) + model_name + '.txt'
    model_filename = output_dir + 'm_' + model_name + '.pt'
    with open(message_filename, 'w') as out:
        out.write('start\n')
    char_vocab_size = len(alph)+5
    global model
    model = torch.load(model_filename)
    model.to(device)

    
    print (args)
    print (message_filename)

    check(X_test, mask_test, Y_test, test_noise_tokens, test_tokens, id2vocab, len(vocab), args.seq_len, args)
    test_acc = evaluate(X_test, mask_test, Y_test, args.batch_size, args.seq_len, len(vocab), args)
    message = ('=' * 89
            + '\n| {} --> {} | test acc {:5.2f}'.format(source, target, test_acc)
            + "\n" +  '=' * 89)
    output_s(message, message_filename)

if __name__=='__main__':
    main()
