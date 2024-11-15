import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser


unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h # number of hidden units (neurons)
        self.W1 = nn.Linear(input_dim, h) # input layer to hidden layer
        self.activation = nn.ReLU() # relu - hidden layer activation function
        self.output_dim = 5 # number of classes
        self.W2 = nn.Linear(h, self.output_dim) # hidden layer to output layer

        self.softmax = nn.LogSoftmax() # softmax - output layer activation function
        self.loss = nn.NLLLoss() # cross-entropy/negative log likelihood loss function

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # [to fill] obtain first hidden layer representation
        hidden = self.W1(input_vector)
        hidden = self.activation(hidden)
        # [to fill] obtain output layer representation
        output = self.W2(hidden)
        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(output)
        return predicted_vector


# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data

# Returns:
# train_data = A list of pairs (document, y) from training data
# test_data = A list of pairs (document, y) from test data 
# valid_data = A list of pairs (document, y) from validation data
def load_data(train_data, val_data, test_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)
    with open(test_data) as test_f:
        test = json.load(test_f)

    tra = []
    tes = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in test:
        tes.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra, tes, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_infer', action='store_true')
    args = parser.parse_args()

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # load data
    print("========== Loading data ==========")
    train_data, test_data, valid_data = load_data(args.train_data, args.val_data, args.test_data) 
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    test_data = convert_to_vector_representation(test_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    

    model = FFNN(input_dim = len(vocab), h = args.hidden_dim)
    optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)
    if args.do_train:
        print("========== Training for {} epochs ==========".format(args.epochs))
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            loss = None
            correct = 0
            total = 0
            start_time = time.time()
            print("Training started for epoch {}".format(epoch + 1))
            random.shuffle(train_data) # Good practice to shuffle order of training data
            minibatch_size = 16 
            N = len(train_data) 
            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                loss = None
                for example_index in range(minibatch_size):
                    input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    correct += int(predicted_label == gold_label)
                    total += 1
                    example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss
                loss = loss / minibatch_size
                loss.backward()
                optimizer.step()
            print("Training completed for epoch {}".format(epoch + 1))
            print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
            print("Training time for this epoch: {}".format(time.time() - start_time))


            loss = None
            correct = 0
            total = 0
            start_time = time.time()
            print("Validation started for epoch {}".format(epoch + 1))
            minibatch_size = 16 
            N = len(valid_data) 
            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                loss = None
                for example_index in range(minibatch_size):
                    input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    correct += int(predicted_label == gold_label)
                    total += 1
                    example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss
                loss = loss / minibatch_size
            print("Validation completed for epoch {}".format(epoch + 1))
            print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
            print("Validation time for this epoch: {}".format(time.time() - start_time))
            
    if args.do_infer:
        print("========== Inference ==========")
        # Take a random sample from test data
        sample_size = 10  # You can adjust this number
        test_samples = random.sample(test_data, sample_size)
        
        model.eval()  # Set model to evaluation mode
        total_loss = 0
        correct = 0
        
        print("\nSample Predictions:")
        print("-" * 50)
        
        with torch.no_grad():  # No need to track gradients during inference
            for input_vector, gold_label in test_samples:
                # Get model prediction
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                
                # Compute loss
                loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                total_loss += loss.item()
                
                # Track accuracy
                correct += int(predicted_label == gold_label)
                
                # Print prediction details
                print(f"Predicted rating: {predicted_label + 1} stars")
                print(f"Actual rating: {gold_label + 1} stars")
                print(f"Sample loss: {loss.item():.4f}")
                print("-" * 50)
        
        # Print summary statistics
        print("\nInference Summary:")
        print(f"Average loss: {total_loss / sample_size:.4f}")
        print(f"Sample accuracy: {correct / sample_size:.2%}")
    
    # write out to results/test.out
    