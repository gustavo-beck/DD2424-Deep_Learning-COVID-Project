# This file is used to define the method that will train and test MemeNet.
from tqdm import tqdm
from memeNet import *
import numpy as np
from sklearn.metrics import classification_report, multilabel_confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt



def pred_acc(original, predicted):
    return torch.round(predicted.cpu()).eq(original.cpu()).sum().numpy(), torch.round(predicted.cpu()).eq(original.cpu()).sum(dim=0).numpy()


def evaluate_model(model, data_loader, data_set, criterion, device, num_labels, labels_names):
    acc_total = 0
    acc_labels = np.zeros((num_labels))
    loss = 0
    model.eval()
    predictions_matrix = np.zeros((1, num_labels))
    labels_matrix = np.zeros((1, num_labels))
    with torch.no_grad():
        for batch_images, batch_labels in tqdm(data_loader):
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            outputs = model(batch_images)
            acc_total_aux, acc_label_aux = pred_acc(batch_labels, outputs)
            acc_total += acc_total_aux
            acc_labels += acc_label_aux
            loss += criterion(outputs.double(), batch_labels.to(device)).item() * len(batch_labels)
            predictions_matrix = np.concatenate((predictions_matrix, outputs.cpu().detach().numpy()), axis=0)
            labels_matrix = np.concatenate((labels_matrix, batch_labels.cpu().detach().numpy()), axis=0)

    predictions_matrix = predictions_matrix[1:]
    labels_matrix = labels_matrix[1:]
    print(classification_report(labels_matrix.astype(int), np.round(predictions_matrix).astype(int), target_names=labels_names))
    correct_total = acc_total / num_labels / len(data_set)
    correct_labels = acc_labels / len(data_set)
    return correct_total, correct_labels, loss / len(data_set)


def train_memeNet(model, trainloader, valloader, testloader, optimizer, scheduler, criterion, device, train_set,
                  val_set, test_set, num_labels, epochs=50):
    labels_names = list(train_set.df.columns)[2:]
    for epoch in range(epochs):
        print(epoch)
        training_samples = 0
        training_acc_total = 0
        training_acc_labels = np.zeros((num_labels))
        model.train()
        predictions_matrix = np.zeros((1, num_labels))
        labels_matrix = np.zeros((1, num_labels))

        for batch_images, batch_labels in tqdm(trainloader):
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            optimizer.zero_grad()  # we set the gradients to zero
            outputs = model(batch_images)  # feed forward
            training_acc_total_aux, training_acc_labels_aux = pred_acc(batch_labels, outputs)
            training_acc_total += training_acc_total_aux
            training_acc_labels += training_acc_labels_aux
            loss = criterion(outputs.double(), batch_labels)  # claculating loss
            loss.backward()  # backpropagate the loss
            optimizer.step()  # updating parameters
            aux = outputs.cpu().detach().numpy()
            aux2 = batch_labels.cpu().detach().numpy()
            predictions_matrix = np.concatenate((predictions_matrix, outputs.cpu().detach().numpy()), axis=0)
            labels_matrix = np.concatenate((labels_matrix, batch_labels.cpu().detach().numpy()), axis=0)

        predictions_matrix = predictions_matrix[1:]
        labels_matrix = labels_matrix[1:]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_labels):
            fpr[i], tpr[i], _ = roc_curve(labels_matrix[:, i], predictions_matrix[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot of a ROC curve for a specific class
        for i in range(num_labels):
            plt.figure()
            plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            title = 'Label_'+str(labels_names[i])+'_epoch_'+str(epoch)
            plt.savefig(title)


        # print(classification_report(labels_matrix.astype(int), np.round(predictions_matrix).astype(int), target_names=labels_names))
        print('Training acc total:', (training_acc_total / num_labels) / len(train_set))
        print('Training acc per label:', (training_acc_labels / len(train_set)))
        # print(roc_auc_score(labels_matrix, predictions_matrix, labels=labels_names, multi_class='ovr'))

        # Compute Validation acc
        print('Computing validation accuracy...')
        val_acc, val_acc_labels, val_loss = evaluate_model(model, valloader, val_set, criterion, device, num_labels, labels_names)
        print('Validation acc total:', val_acc)
        print('Validation acc per label:', val_acc_labels)
        scheduler.step(val_loss)  # update learning rate

    print('Computing test accuracy...')
    test_acc, test_acc_labels, test_loss = evaluate_model(model, testloader, test_set, criterion, device, num_labels, labels_names)
    print('Test acc total:', test_acc)
    print('Test acc per label:', test_acc_labels)
    return model


