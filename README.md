# Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels Implementation

### PAPER:
[Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels ](https://arxiv.org/abs/1804.06872)

### Model:
```python
class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.1):
        self.dropout_rate = dropout_rate
        super(CNN, self).__init__()
        self.c1=nn.Conv2d(input_channel,32,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv2d(32,64,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv2d(64,128,kernel_size=3,stride=1, padding=1)
        self.c4=nn.Conv2d(128,256,kernel_size=3,stride=1, padding=1)
        self.l_c1=nn.Linear(256,n_outputs)
        self.bn1=nn.BatchNorm2d(32)
        self.bn2=nn.BatchNorm2d(64)
        self.bn3=nn.BatchNorm2d(128)
        self.bn4=nn.BatchNorm2d(256)

    def forward(self, x,):
        h=x
        h=self.c1(h)
        h=F.leaky_relu(self.call_bn(self.bn1, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=self.c2(h)
        h=F.leaky_relu(self.call_bn(self.bn2, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=self.c3(h)
        h=F.leaky_relu(self.call_bn(self.bn3, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=self.c4(h)
        h=F.leaky_relu(self.call_bn(self.bn4, h), negative_slope=0.01)
        h=F.avg_pool2d(h, kernel_size=h.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        logit=self.l_c1(h)
        return logit

    @staticmethod
    def call_bn(bn, x):
        return bn(x)

```


## Train:

```python

def train(train_loader, test_loader, epoch_idx, model1, optimizer1, model2, optimizer2):
    """Train the two models using Co-teaching method"""
    print(f'epoch {epoch_idx+1}/{n_epoch}')
    remember_rate = 1 - max(forget_rate, forget_rate * (epoch_idx/epoch_decay_start))
    for batch_idx, (images, labels, _) in enumerate(train_loader):
        num_remember = int(remember_rate * labels.shape[0])
        images = images.to(device)
        labels = labels.to(device)
        logits1 = model1(images)
        logits2 = model2(images)
        loss_1, loss_2 = loss_coteaching(logits1, logits2, labels, num_remember)
        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
    accuracy1, accuracy2 = evaluate(test_loader, model1, model2)
    print(f"accuracy 1: {accuracy1*100}\naccuracy 2: {accuracy2*100}")

def loss_coteaching(y_1, y_2, t, num_remember):
    """Co-teaching loss exchange"""
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_1_sorted = torch.argsort(loss_1.data)
    ind_2_sorted = torch.argsort(loss_2.data)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]

    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember

```


## Symmetric noise:

Noise Matrix:

```
 [[0.8        0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.02222222]
 [0.02222222 0.8        0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.02222222]
 [0.02222222 0.02222222 0.8        0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.02222222]
 [0.02222222 0.02222222 0.02222222 0.8        0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.02222222]
 [0.02222222 0.02222222 0.02222222 0.02222222 0.8        0.02222222 0.02222222 0.02222222 0.02222222 0.02222222]
 [0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.8        0.02222222 0.02222222 0.02222222 0.02222222]
 [0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.8        0.02222222 0.02222222 0.02222222]
 [0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.8        0.02222222 0.02222222]
 [0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.8        0.02222222]
 [0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.02222222 0.8       ]]
```

## Results:
```python
epoch 1/100
accuracy of model 1 on test set: 57.08000063896179
accuracy of model 2 on test set: 58.21999907493591
epoch 2/100
accuracy of model 1 on test set: 61.76999807357788
accuracy of model 2 on test set: 62.61000037193298
epoch 3/100
accuracy of model 1 on test set: 62.909996509552
accuracy of model 2 on test set: 65.93999862670898
epoch 4/100
accuracy of model 1 on test set: 66.76999926567078
accuracy of model 2 on test set: 68.11000108718872
epoch 5/100
accuracy of model 1 on test set: 66.5399968624115
accuracy of model 2 on test set: 69.26999688148499
epoch 6/100
accuracy of model 1 on test set: 66.839998960495
accuracy of model 2 on test set: 70.21999955177307
epoch 7/100
accuracy of model 1 on test set: 65.66999554634094
accuracy of model 2 on test set: 69.5199966430664
epoch 8/100
accuracy of model 1 on test set: 69.87000107765198
accuracy of model 2 on test set: 70.91000080108643
epoch 9/100
accuracy of model 1 on test set: 70.45999765396118
accuracy of model 2 on test set: 70.92999815940857
epoch 10/100
accuracy of model 1 on test set: 70.03999948501587
accuracy of model 2 on test set: 70.94999551773071
epoch 11/100
accuracy of model 1 on test set: 71.079999100845032
accuracy of model 2 on test set: 71.0599958896637
.
.
.
```
