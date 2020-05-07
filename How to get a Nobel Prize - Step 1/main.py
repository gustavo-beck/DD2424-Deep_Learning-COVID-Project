import glob
from dataLoading import *
from memeNet_train import *
from memeNet import *
import pandas as pd
from natsort import natsorted  # This library sorts a list in a "natural" way


def cure_covid(path, images_df, image_size, net_num, WEIGHTS=False):
    WEIGHTS = False
    # TODO READ THE IMAGES_DF TO CREATE THE DICTIONARY WITH TRAIN, VAL AND TEST

    labels = ['train', 'val', 'test']
    directories = {}

    diseases = np.unique(
        len(images_df.columns) - 2)  # We compute the number of diseases (labels) by computing the length
    # of the data frame and subtracting two columns (one column which is the name of the images and another which is
    # whether it's train, val or test)
    idx = np.arange(len(diseases))
    class_dict = dict(zip(diseases, idx))

    # Define transformations
    our_transforms = torchvision.transforms.ToTensor()

    # Create data sets
    train_set = CustomDatasetFromImages(directories['train'], class_dict, transforms=our_transforms)
    val_set = CustomDatasetFromImages(directories['val'], class_dict, transforms=our_transforms)
    test_set = CustomDatasetFromImages(directories['test'], class_dict, transforms=our_transforms)

    # Create data loaders
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=10, num_workers=0, shuffle=False)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=10, num_workers=0, shuffle=False)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=10, num_workers=0, shuffle=False)

    # Create device to perform computations in GPU (if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Compute weights for class imbalance if required
    if WEIGHTS:
        print('Calculating frequencies..')
        freq = np.zeros(len(class_dict.keys()))
        for sample in train_set:
            freq[sample[1]] += 1
        weights = len(train_set) / freq / len(class_dict.keys())
        class_weights = torch.FloatTensor(weights).to(device)

    # Create Model
    methods = {
        18: ResNet18,
        34: ResNet34,
        50: ResNet50,
        101: ResNet101,
        152: ResNet152
    }
    if net_num in methods:
        model = methods[net_num](num_classes=len(diseases))  # + argument list of course
    else:
        raise Exception("Method %s not implemented" % net_num)

    # model = ResNet34(num_classes=len(diseases))

    # Modify last FC layer to be suitable for multi-lable classification (adding sigmoids)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Linear(num_ftrs, len(diseases)), nn.Sigmoid())
    model.to(device)

    if WEIGHTS:
        criterion = torch.nn.BCELoss(weight=class_weights)

    else:
        criterion = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

    # scheduler is used to adjust the learning rate,
    # this oparticular scheduler uses the validation accuracy to adjust the learning rate,
    # other schedulers dont require the validation accuracy
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, eps=1e-06)
    model = train_memeNet(model, trainloader, valloader, testloader, optimizer, scheduler, criterion, device, train_set,
                          val_set, test_set, epochs=50)

def main():
    path = ''
    image_df = pd.read_csv('organized_dataset.csv')
    image_size = 224
    network = 18 # 18, 34, 50, 101 or 152 depending on the desired resnet
    cure_covid(path, image_df, image_size, network)


if __name__ == "__main__":
    main()
