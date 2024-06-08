# MNIST-Digit-Classification-with-CNN

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for classifying digits from the MNIST dataset. The network architecture and training process are defined using PyTorch, and the dataset is loaded using `torchvision.datasets`.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training and Testing](#training-and-testing)
- [Results](#results)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/mnist-cnn.git
   cd mnist-cnn
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

To train and test the model, run the following script:
```sh
python mnist_cnn.py
```

## Model Architecture

The CNN architecture consists of the following layers:
- Convolutional Layer 1: 1 input channel, 10 output channels, kernel size of 5
- Convolutional Layer 2: 10 input channels, 20 output channels, kernel size of 5
- Dropout Layer
- Fully Connected Layer 1: 320 input features, 50 output features
- Fully Connected Layer 2: 50 input features, 10 output features

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.softmax(x)
```

## Training and Testing

The training process is defined in the `train` function, which optimizes the model using the Adam optimizer and the Cross Entropy loss function. The testing process is defined in the `test` function, which evaluates the model's performance on the test dataset.

```python
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 20 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders['train'].dataset)} ({100. * batch_idx / len(loaders['train']):.0f}%)]\t{loss.item():.6f}")

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(loaders['test'].dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders['test'].dataset)} ({100. * correct / len(loaders['test'].dataset):.0f}%)\n")
```

## Results

After training for 10 epochs, the model achieves a high accuracy on the MNIST test set. The training and testing accuracy and loss are printed during the training process.

## Visualization

The trained model can be used to make predictions on individual images from the MNIST dataset. The following code snippet shows how to visualize a prediction:

```python
import matplotlib.pyplot as plt

model.eval()
data, target = test_data[0]
data = data.unsqueeze(0).to(device)
output = model(data)
prediction = output.argmax(dim=1, keepdim=True).item()
print(f'Prediction: {prediction}')
image = data.squeeze(0).squeeze(0).cpu().numpy()
plt.imshow(image, cmap='gray')
plt.show()
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Feel free to modify the content to better match your specific project details and requirements.
