# This file is used to define the method that will train and test MemeNet.
from tqdm import tqdm
from memeNet import *



def pred_acc(original, predicted):
    return torch.round(predicted).eq(original).sum().numpy()


def evaluate_model(model, data_loader, data_set, criterion, device):
    acc = 0
    loss = 0
    model.eval()
    with torch.no_grad():
        for batch_images, batch_labels in data_loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            outputs = model(batch_images)
            acc += pred_acc(batch_labels, outputs)
            loss += criterion(outputs, batch_labels.long().to(device)).data[0] * len(batch_labels)

    correct = acc / len(data_set.labels_dict) / len(data_set)
    return correct, loss / len(data_set)


def train_memeNet(model, trainloader, valloader, testloader, optimizer, scheduler, criterion, device, train_set,
                  val_set, test_set, epochs=50):
    for epoch in range(epochs):
        print(epoch)
        training_samples = 0
        training_acc = 0
        model.train()
        for batch_images, batch_labels in tqdm(trainloader):
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            optimizer.zero_grad()  # we set the gradients to zero
            outputs = model(batch_images)  # feed forward
            training_acc += pred_acc(batch_labels, outputs)
            loss = criterion(outputs, batch_labels.long().to(device))  # claculating loss
            loss.backward()  # backpropagate the loss
            optimizer.step()  # updating parameters

        print('Training acc:', (training_acc / len(train_set.labels_dict)) / len(train_set))

        # Compute Validation acc
        val_acc, val_loss = evaluate_model(model, valloader, val_set, criterion, device)
        print('Validation acc:', val_acc)
        scheduler.step(val_loss)  # update learning rate

    print('Computing test accuracy...')
    test_acc, test_loss = evaluate_model(model, testloader, test_set, criterion, device)
    print('Test acc:', test_acc)
    return model
