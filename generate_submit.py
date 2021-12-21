from model import *
import csv


batch_size = 10
labels = []

test_loader = LoadSignalDataset("./testA.csv", need_over_sample=False, test=True)
test_loader = DataLoader(test_loader, batch_size=batch_size, drop_last=False, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('Complex_CNN-unbalanced.pth').to(device)

with torch.no_grad():
    for test_data in test_loader:
        test_data = test_data.to(device)
        model.eval()
        test_output = model(test_data.to(torch.float32))
        test_labels = test_output.argmax(dim=1).unsqueeze(dim=1)
        labels += torch.zeros(batch_size, 4, dtype=torch.int8).to(device).scatter_(1, test_labels, 1).tolist()

    index = 100000
    for i in range(len(labels)):
        labels[i] = [index] + labels[i]
        index += 1
    with open("submit.csv", "w") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["id", "label_0", "label_1", "label_2", "label_3"])
        writer.writerows(labels)
