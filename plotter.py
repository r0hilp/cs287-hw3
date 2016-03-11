import matplotlib.pyplot as plt

train = []
valid = []
with open('train_numbers', 'r') as f_train:
    for line in f_train:
        train.append(float(line.strip()))

with open('valid_numbers', 'r') as f_valid:
    for line in f_valid:
        valid.append(float(line.strip()))

plt.figure()
plt.title('Training NNLM')
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
valid_plot,  = plt.plot(valid, 'b+', label='Valid')
train_plot, = plt.plot(train, 'r+', label='Train')
plt.legend(handles=[valid_plot, train_plot])
plt.show()
