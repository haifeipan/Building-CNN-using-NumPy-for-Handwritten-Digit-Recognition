import os
import numpy as np
import gzip

def softmax_f(x):
    prob = np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)
    return prob

def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]
def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError(
          'Invalid magic number %d in MNIST image file: %s' %
          (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data
def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot
def extract_labels(filename, one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    if one_hot:
      return dense_to_one_hot(labels)
    return labels
def read_data_sets(train_dir, one_hot=False, dtype=np.float32):
  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
  VALIDATION_SIZE = 0
  local_file = os.path.join(train_dir, TRAIN_IMAGES)
  train_images = extract_images(local_file)
  local_file = os.path.join(train_dir, TRAIN_LABELS)
  train_labels = extract_labels(local_file, one_hot=one_hot)
  local_file = os.path.join(train_dir, TEST_IMAGES)
  test_images = extract_images(local_file)
  local_file = os.path.join(train_dir, TEST_LABELS)
  test_labels = extract_labels(local_file, one_hot=one_hot)
  validation_images = train_images[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]
  train = [train_images / 255.0, train_labels]
  valid = [validation_images / 255.0, validation_labels]
  test = [test_images / 255.0, test_labels]
  return train, valid, test

class cnn_layer():
    def __init__(self, name, kernel, inc, outc):
        super(cnn_layer, self).__init__()
        self.name = name
        self.kernel = kernel
        self.inc = inc
        self.outc = outc
        self.weight = np.random.randn(kernel, kernel, inc, outc) * np.sqrt(2.0 / (kernel * kernel * inc))
        self.bias = np.zeros(outc)
        self.weight_diff = 0
        self.bias_diff = 0

    def forward(self, x):
        self.x = x
        k = self.kernel
        n, h, w, c = x.shape
        h_out = h - (k - 1)
        w_out = w - (k - 1)
        weight = self.weight.reshape(-1, self.outc)
        output = np.zeros((n, h_out, w_out, self.outc))
        for i in range(h_out):
            for j in range(w_out):
                inp = x[:, i:i+k, j:j+k, :].reshape(n, -1)
                out = inp.dot(weight) + self.bias
                output[:, i, j, :] = out.reshape(n, -1)
        return output

    def backward(self, diff):
        n, h, w, c = diff.shape
        k = self.kernel
        h_in = h + (k - 1)
        w_in = w + (k - 1)

        weight_diff = np.zeros((k, k, self.inc, self.outc))
        for i in range(k):
            for j in range(k):
                #inp = (n, 28, 28, c) => (n*28*28, c) => (c, n*28*28)
                inp = self.x[:, i:i+h, j:j+w, :].reshape(-1, self.inc).T
                #diff = n, 28, 28, 6 => (n*28*28, 6)
                diff_out = diff.reshape(-1, self.outc)
                weight_diff[i, j, :, :] = inp.dot(diff_out)
        bias_diff = np.sum(diff, axis=(0, 1, 2))

        pad = k - 1
        diff_pad = np.pad(diff, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
        rotated_weight = self.weight[::-1, ::-1, :, :].transpose(0, 1, 3, 2).reshape(-1, self.inc)
        back_diff = np.zeros((n, h_in, w_in, self.inc))
        for i in range(h_in):
            for j in range(w_in):
                diff_out = diff_pad[:, i:i+k, j:j+k, :].reshape(n, -1)
                out = diff_out.dot(rotated_weight)
                back_diff[:, i, j, :] = out.reshape(n, -1)

        self.weight_diff = momentum * self.weight_diff + (1 - momentum) * weight_diff
        self.bias_diff = momentum * self.bias_diff + (1 - momentum) * bias_diff
        self.weight -= learning_rate * self.weight_diff + weight_decay * self.weight
        self.bias -= learning_rate * self.bias_diff + weight_decay * self.bias

        return back_diff
    
class MaxPooling_layer():
    def forward(self, x):
        n, h, w, c = x.shape
        x_grid = x.reshape(n, h // 2, 2, w // 2, 2, c)
        out = np.max(x_grid, axis=(2, 4))
        self.mask = (out.reshape(n, h // 2, 1, w // 2, 1, c) == x_grid)
        return out

    def backward(self, diff):
        n, h, w, c = diff.shape
        diff_grid = diff.reshape(n, h, 1, w, 1, c)
        return (diff_grid * self.mask).reshape(n, h * 2, w * 2, c)

class relu_layer():
    def __init__(self):
        self.judge=None
    def forward(self,x):
        self.judge=(x<=0)
        out=x.copy()
        out[self.judge]=0
        return out
    def backward(self,diff):
        diff[self.judge]=0
        return diff

class fc_layer():
    def __init__(self, name, inc, outc):
        super(fc_layer, self).__init__()
        self.name = name
        self.weight = np.random.randn(inc, outc) * np.sqrt(2.0 / inc)
        self.bias = np.zeros(outc)
        self.weight_diff = 0
        self.bias_diff = 0

    def forward(self, x):
        self.origin_shape = x.shape
        if x.ndim == 4:
            x = x.reshape(x.shape[0], -1)
        self.x = x
        return x.dot(self.weight) + self.bias
   
    def backward(self, diff):
        weight_diff = self.x.T.dot(diff)
        bias_diff = np.sum(diff, axis=0)
        back_diff = diff.dot(self.weight.T).reshape(self.origin_shape)

        self.weight_diff = momentum * self.weight_diff + (1 - momentum) * weight_diff
        self.bias_diff = momentum * self.bias_diff + (1 - momentum) * bias_diff
        self.weight -= learning_rate * self.weight_diff + weight_decay * self.weight
        self.bias -= learning_rate * self.bias_diff + weight_decay * self.bias

        return back_diff

class softmax_layer():
    def forward(self, x):
        softmax = softmax_f(x)
        self.softmax = softmax
        output = np.argmax(softmax, axis=1)
        if not hasattr(self, 'y'):
            return output

        y = self.y
        label = np.argmax(y, axis=1)
        loss = -np.sum(y * np.log(softmax) + (1 - y) * np.log(1 - softmax)) / len(y)
        accuracy = np.sum(output==label) / float(len(label))
        return loss, accuracy

    def backward(self, diff):
        return self.softmax - self.y

    def set_label(self, label):
        self.y = label

class LeNet:
    def __init__(self,
                 input_hight=1,
                 conv_param={'filter_num1':6, 'kernel1':5,'filter_num2':16, 'kernel2':5},
                 fc_parm = {'hidden_size1':120,'hidden_size2':84, 'output_size':10}):
        conv1 = cnn_layer("conv1", conv_param['kernel1'], input_hight, conv_param['filter_num1'])
        relu1 = relu_layer()
        pool1 = MaxPooling_layer()
        conv2 = cnn_layer("conv2", conv_param['kernel2'], conv_param['filter_num1'], conv_param['filter_num2'])
        relu2 = relu_layer()
        pool2 = MaxPooling_layer()
        # fc3 = FC("fc3", 400, fc_parm['hidden_size1'])
        fc3 = fc_layer("fc3", 256, fc_parm['hidden_size1'])
        relu3 = relu_layer()
        fc4 = fc_layer("fc4", fc_parm['hidden_size1'], fc_parm['hidden_size2'])
        relu4 = relu_layer()
        fc5 = fc_layer("fc5", fc_parm['hidden_size2'], fc_parm['output_size'])
        loss = softmax_layer()
        self.layers = [conv1, relu1, pool1, conv2, relu2, pool2, fc3, relu3, fc4, relu4, fc5, loss]


    def train(self,images,labels):

        index = 0
        num_images = len(images)
        iterations = int(num_images/batch_size)
        total_N = int((num_images/batch_size)*epochs)
        print("训练样本的大小为 %d, batch_size %d, epochs %d, 总训练次数 %d" % (num_images, batch_size, epochs, total_N))
        print('训练开始：')
        for i in range(total_N):
            x = images[index:index + batch_size] #mini batch sgd
            y = labels[index:index + batch_size]
            index += batch_size
            index = index % num_images

            loss = self.layers[-1]
            loss.set_label(y)

            for layer in self.layers:
                x = layer.forward(x)
            # print("step %d: loss=%.6f, accuracy=%.4f, lr=%g" % (i, x[0], x[1], learning_rate))
            epoch = (i//iterations) + 1
            iteration = (i+1) % iterations
            print("epochs %d, iterations %d, total_step %d, loss %.4f, accuracy_rate %.6f" % (epoch, iteration, i+1, x[0], x[1]))
            diff = 1.0
            for layer in reversed(self.layers):
                diff = layer.backward(diff)
            # learning_rate *= (1 - learning_rate_decay)

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def save_parameter(self, file_name):
        model = {}
        model['conv1'] = {'w':self.layers[0].weight,'b':self.layers[0].bias}
        model['conv2'] = {'w':self.layers[3].weight,'b':self.layers[3].bias}
        model['fc3'] = {'w':self.layers[6].weight,'b':self.layers[6].bias}
        model['fc4'] = {'w':self.layers[8].weight,'b':self.layers[8].bias}
        model['fc5'] = {'w':self.layers[10].weight,'b':self.layers[10].bias}
        # for layer in self.layers:
        #     if isinstance(layer, Trainable):
        #         model[layer.name] = {"w": layer.weight, "b": layer.bias}
        np.save(file_name, model)

    def load_parameter(self, file_name):
        model = np.load(file_name, allow_pickle=True).item()
        self.layers[0].weight = model[self.layers[0].name]['w']
        self.layers[0].bias = model[self.layers[0].name]['b']

        self.layers[3].weight = model[self.layers[3].name]['w']
        self.layers[3].bias = model[self.layers[3].name]['b']

        self.layers[6].weight = model[self.layers[6].name]['w']
        self.layers[6].bias = model[self.layers[6].name]['b']

        self.layers[8].weight = model[self.layers[8].name]['w']
        self.layers[8].bias = model[self.layers[8].name]['b']

        self.layers[10].weight = model[self.layers[10].name]['w']
        self.layers[10].bias = model[self.layers[10].name]['b']


        # for layer in self.layers:
        #     if isinstance(layer, Trainable):
        #         layer.weight = model[layer.name]["w"]
        #         layer.bias = model[layer.name]["b"]

if __name__ == '__main__':
    train_dir = './mnist'
    train, valid, test = read_data_sets(train_dir, one_hot=True, dtype=np.float32)
    x = train[0]
    y = train[1]
    lenet = LeNet(input_hight=1,
                 conv_param={'filter_num1':6, 'kernel1':5,'filter_num2':16, 'kernel2':5},
                 fc_parm = {'hidden_size1':120,'hidden_size2':84, 'output_size':10})
    # 在这里输入参数的设置
    learning_rate = 0.0002
    momentum = 0.95    # 选择momentum作为优化方法
    batch_size = 1000
    epochs = 12
    weight_decay = 0.001

    lenet.train(x, y)
    print('训练完毕！已经将本次训练结果的参数保存至"LeNet_result.npy"文件')   # 保存的是W和B的值
    lenet.save_parameter("LeNet_result.npy")


    net = LeNet()
    net.load_parameter("LeNet_result.npy")
    x = test[0]
    y_onehot = test[1]
    y = np.array([np.argmax(item) for item in y_onehot]) # 把onehot转化为一维标签
    predict = net.predict(x)
    accuracy = np.sum(predict == y)
    accuracy_rate = float(accuracy) / len(y)
    print('以下是在测试集的表现')
    print("accuracy_rate=%f" % accuracy_rate)
    