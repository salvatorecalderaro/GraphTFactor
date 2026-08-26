import torch
import torch.nn as nn
from torch_geometric.profile import timeit
from tqdm import tqdm
import torch.optim as optim
import numpy as np

"""
Kim, G. B., Gao, Y., Palsson, B. O., & Lee, S. Y. (2021). 
DeepTFactor: A deep learning-based tool for the prediction of transcription factors. 
Proceedings of the National Academy of Sciences, 118(2), e2021171118.
"""
class DeepTFactor(nn.Module):
    """
    Initialize the DeepTFactor model.

    Parameters:
    - out_features (list): A list of integers representing the number of output features for each layer.
    If not provided, a default list of zeros is used.

    Attributes:
    - explainECs (list): A list of integers representing the number of output features for each layer.
    - layer_info (list): A list of lists containing the kernel sizes for each subnetwork in the CNN.
    - cnn0 (CNN): The first convolutional neural network layer.
    - fc1 (nn.Linear): The first fully connected layer.
    - bn1 (nn.BatchNorm1d): The first batch normalization layer.
    - fc2 (nn.Linear): The second fully connected layer.
    - bn2 (nn.BatchNorm1d): The second batch normalization layer.
    - out_act (nn.Sigmoid): The output activation function.
    - relu (nn.ReLU): The activation function used in the CNN.

    Returns:
    None. The function initializes the model attributes and does not return a value.
    """
    def __init__(self, out_features=[0]):
        super(DeepTFactor, self).__init__()
        self.explainECs = out_features
        self.layer_info = [[4, 4, 16], [12, 8, 4], [16, 4, 4]]
        self.cnn0 = CNN(self.layer_info)
        self.fc1 = nn.Linear(in_features=128*3, out_features=512)
        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=len(out_features))
        self.bn2 = nn.BatchNorm1d(num_features=len(out_features))
        self.out_act = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.init_weights()


    def init_weights(self):
        """
        Initialize the weights of the neural network layers.

        This function initializes the weights of the convolutional and fully connected layers in the DeepTFactor model. It uses the Xavier uniform initialization method, which is a common technique for initializing weights in neural networks. This method helps to prevent the vanishing and exploding gradient problems that can occur during training.

        Parameters:
        - None

        Returns:
        - None. The function initializes the weights and does not return a value.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        
    def forward(self, x):
        """
        Apply the forward pass of the DeepTFactor model to the input tensor x.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (batch_size, 1, 21, 128).

        Returns:
        - out (torch.Tensor): The output tensor of shape (batch_size, num_output_features) after passing through the DeepTFactor model.
        """
        x = self.cnn0(x)
        x = x.view(-1, 128 * 3)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.out_act(self.bn2(self.fc2(x)))
        return x


class CNN(nn.Module):
    '''
    Use second level convolution.
    channel size: 4 -> 16 
                  8 -> 12
                  16 -> 4
    '''
    def __init__(self, layer_info):
        """
        Initialize the CNN layer for the DeepTFactor model.

        Parameters:
        - layer_info (List[List[int]]): A list of lists containing the kernel sizes for each subnetwork in the CNN.

        Attributes:
        - relu (nn.ReLU): The activation function used in the CNN.
        - dropout (nn.Dropout): The dropout layer with a probability of 0.1.
        - layers (nn.ModuleList): A list of CNN subnetworks created using the make_subnetwork method.
        - conv (nn.Conv2d): The convolutional layer with input channels equal to the sum of channels from all subnetworks and output channels equal to 128*3.
        - batchnorm (nn.BatchNorm2d): The batch normalization layer with 128*3 features.
        - pool (nn.MaxPool2d): The max pooling layer with a kernel size of (1000+pooling_size,1) and a stride of 1.

        Returns:
        None. The function initializes the CNN layer and does not return a value.
        """
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

        self.layers = nn.ModuleList()
        pooling_sizes = []
        for subnetwork in layer_info:
            pooling_size = 0
            self.layers += [self.make_subnetwork(subnetwork)]
            for kernel in subnetwork:
                pooling_size += (-kernel + 1)
            pooling_sizes.append(pooling_size)

        if len(set(pooling_sizes)) != 1:
            raise "Different kernel sizes between subnetworks"
        pooling_size = pooling_sizes[0]
        num_subnetwork = len(layer_info)

        self.conv = nn.Conv2d(in_channels=128*num_subnetwork, out_channels=128*3, kernel_size=(1,1))
        self.batchnorm = nn.BatchNorm2d(num_features=128*3)
        self.pool = nn.MaxPool2d(kernel_size=(1000+pooling_size,1), stride=1)


    def make_subnetwork(self, subnetwork):
        """
        Create a subnetwork of convolutional layers for the CNN in the DeepTFactor model.

        Parameters:
        - subnetwork (List[int]): A list of integers representing the kernel sizes for each convolutional layer in the subnetwork.

        Returns:
        - subnetwork (nn.Sequential): A subnetwork of convolutional layers as a nn.Sequential object.

        This function creates a subnetwork of convolutional layers for the CNN in the DeepTFactor model. It takes a list of integers representing the kernel sizes for each convolutional layer in the subnetwork as input. The function then creates a subnetwork of convolutional layers using the provided kernel sizes and returns it as a nn.Sequential object.
        """
        subnetworks = []
        for i, kernel in enumerate(subnetwork):
            if i == 0:
                subnetworks.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(kernel, 21)),
                        nn.BatchNorm2d(num_features=128),
                        nn.ReLU(),
                        nn.Dropout(p=0.1)
                    )
                )
            else:
                subnetworks.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(kernel, 1)),
                        nn.BatchNorm2d(num_features=128),
                        nn.ReLU(),
                        nn.Dropout(p=0.1)
                    )
                )
        return nn.Sequential(*subnetworks)

        
    def forward(self, x):
        """
        Apply the forward pass of the DeepTFactor model to the input tensor x.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (batch_size, 1, 21, 128).

        Returns:
        - out (torch.Tensor): The output tensor of shape (batch_size, num_output_features) after passing through the DeepTFactor model.
        """
        xs = []
        for layer in self.layers:
            xs.append(layer(x))
        x = torch.cat(xs, dim=1)
        x = self.relu(self.batchnorm(self.conv(x)))
        x = self.pool(x)
        return x

def train_net(device, net, trainloader, epochs, lr):
    """
    Train the given neural network model on the provided training dataset for a specified number of epochs.

    Parameters:
    - device (torch.device): The device on which the model will be trained.
    - net (DeepTFactor): The neural network model to be trained.
    - trainloader (torch.utils.data.DataLoader): The training dataset loader.
    - epochs (int): The number of epochs for which the model will be trained.
    - lr (float): The learning rate for the Adam optimizer.

    Returns:
    - net (DeepTFactor): The trained neural network model.
    - elapsed_time (float): The time taken to train the model in seconds.

    This function trains the given neural network model on the provided training dataset for a specified number of epochs. It uses the Adam optimizer with the specified learning rate. The function returns the trained neural network model and the time taken to train the model in seconds.
    """
    net.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    net = net.float()
    print("Training..")
    with timeit() as time_counter:
        for epoch in tqdm(range(epochs)):
            net.train(True)
            running_loss = 0
            for i, data in enumerate(trainloader):
                inputs, labels = data
                inputs, labels = data[0].float().to(device), data[1].to(device)
                inputs = inputs.unsqueeze(0).permute(1, 0, 2, 3)
                labels = labels.view(-1, 1).float()
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            running_loss /= (i + 1)
            #print('Epoch {}, loss {:.4f}'.format(epoch + 1, running_loss))
    elapsed_time = time_counter.duration
    return net, elapsed_time


def predict(device,net,testloader):
    """"
    Predict the output of the given neural network model on the provided testing dataset.

    Parameters:
    - net (DeepTFactor): The neural network model to be used for prediction.
    - device (torch.device): The device on which the model will be evaluated.
    - testloader (torch.utils.data.DataLoader): The testing dataset loader.

    Returns:
    - y_test (numpy.ndarray): The true labels of the testing dataset.
    - y_pred (numpy.ndarray): The predicted labels of the testing dataset.
    - proba (numpy.ndarray): The predicted probabilities of the testing dataset.

    This function predicts the output of the given neural network model on the provided testing dataset. It evaluates the model on the testing dataset and returns the true labels, predicted labels, and predicted probabilities of the testing dataset.
    """
    y_test, y_pred, proba = [], [], []
    with torch.no_grad():
        net.eval()
        for x, y in testloader:
            x = x.float().to(device)
            x = x.unsqueeze(0).permute(1, 0, 2, 3)
            y = y.view(-1, 1)
            y_test += y.tolist()
            output = net(x)
            prob = output.cpu()
            pred = torch.round(prob)
            y_pred += pred.tolist()
            proba.extend(prob.tolist())

    y_test = np.array(y_test).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    proba = np.array(proba).reshape(-1)
    return y_test, y_pred, proba