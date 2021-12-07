# Тестовое задание
## Подготовка датасета
В качестве датасета был выбран CIFAR100, и подготовлен класс для работы с [ним](dataset.py):
```
class Cifar100(Dataset):  
    def __init__(self, root='CIFAR100/TRAIN'):  
        self.root = root  
        self.list_of_data = []  
  
        for root, _, files in os.walk(self.root):  
            for file in files:  
                self.list_of_data.append(os.path.join(root.split('/')[-1], file))  
  
        self.classes = os.listdir(self.root)  
  
    def __len__(self):  
        return len(self.list_of_data)  
  
    def __getitem__(self, item):  
        path = self.list_of_data[item]  
        label = path.split('/')[0]  
        label = self.classes.index(label)  
  
        img = cv2.imread(os.path.join(self.root, path))  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
  
        img = img.transpose(-1, 0, 1)  # from numpy to torch image  
  
 return img, label
```

Затем из-за неудачного подбора моеделей перешёл на MNIST

```

class Mnist(Dataset):  
    def __init__(self, images='train-images.idx3-ubyte', labels="train-labels.idx1-ubyte"):  
  
        with open(labels, 'rb') as fimg:  
            magic_nr, _ = struct.unpack(">II", fimg.read(8))  
            lbl = array("b", fimg.read())  
  
        with open(images, 'rb') as fimg:  
            magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))  
            img = array("B", fimg.read())  
  
        ind = [k for k in range(size) if lbl[k] in np.arange(10)]  
  
        images = np.zeros((size, rows, cols), dtype=np.uint8)  
        labels = np.zeros((size, 1), dtype=np.int8)  
  
        for i in range(size):  
            images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))  
            labels[i] = lbl[ind[i]]  
  
        self.images = images  
        self.labels = labels.squeeze()  
  
    def __len__(self):  
        return len(self.labels)  
  
    def __getitem__(self, item):  
  
        img = self.images[item].reshape(1, 28, 28)  
        label = self.labels[item]  
  
        return img, label
```

## Модель автоенкодера
Сделана простая модель [автоенкодера](model.py). Тренировал 250 эпох, до такого результата:
![ae_result.png](images/ae_result.png)
Модель Unet на mnist  получила такой результат:
![ae_result.png](images/mnist_result.png)

## Модель Классификатора
Модель классифицирует объекты после енкодера. 

|top 3 | top 5 | top 10 |
--- | --- | ---
|0.2429|0.3141 |0.4395|

Качество модели на Unet:

|top 1 | top 3 | top 5 |
--- | --- | ---
|0.97|0.98 |1|


Распределение 10ти классов на Cifar100

![ae_result.png](images/two-dimensional.png)

Распределение классов на Mnist

![ae_result.png](images/two_dim_mnist.png)

