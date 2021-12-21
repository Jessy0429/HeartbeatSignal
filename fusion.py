from model import *
import csv


models = ['Complex_CNN-unbalanced.pth',  'Res_CNN-unbalanced.pth', 'Res_CNN1-unbalanced.pth', 'SE_CNN-unbalanced.pth', 'SE_Res_CNN-unbalanced.pth']
weights = [1.2, 0.9, 1.1, 0.9, 0.9]
batch_size = 10


test_loader = LoadSignalDataset("./testA.csv", need_over_sample=False, test=True)
test_loader = DataLoader(test_loader, batch_size=batch_size, drop_last=False, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    uninit_weighted_labels = True

    for model_path, weight in zip(models, weights):
        model = torch.load(model_path).to(device)
        uninit_labels = True
        weight = torch.tensor(weight).to(device)

        for test_data in test_loader:
            test_data = test_data.to(device)
            model.eval()
            test_output = model(test_data.to(torch.float32))
            if uninit_labels:
                labels = test_output
                uninit_labels = False
            else:
                labels = torch.cat((labels, test_output), dim=0)

        labels = labels * weight

        if uninit_weighted_labels:
            weighted_labels = labels
            uninit_weighted_labels = False
        else:
            weighted_labels += labels

    test_labels = weighted_labels.argmax(dim=1).unsqueeze(dim=1)
    labels = torch.zeros(20000, 4, dtype=torch.int8).to(device).scatter_(1, test_labels, 1).tolist()


    index = 100000
    for i in range(len(labels)):
        labels[i] = [index] + labels[i]
        index += 1
    with open("submit.csv", "w") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["id", "label_0", "label_1", "label_2", "label_3"])
        writer.writerows(labels)