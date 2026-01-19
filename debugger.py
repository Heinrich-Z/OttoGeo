import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import json
import os

from custom_dataset import CustomDataset, DataTransform
from modulus import LeNet, AlexNet

def main(case_name, data_name):
    # mps for Apple M chips
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)

    # hyperparameter loading
    cfg_path = './conf/config.json'
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    # Testing
    if case_name == 'AlexNet':
        best_model = AlexNet(
            in_channels=cfg['arch'][case_name]['in_channels'],
            num_classes=cfg['arch'][case_name]['num_classes'],
        ).to(device)

    elif case_name == 'LeNet':
        best_model = LeNet(
            in_channels=cfg['arch'][case_name]['in_channels'],
            num_classes=cfg['arch'][case_name]['num_classes'],
        ).to(device)

    else:
        best_model = None
        return

    run_dir = 'runs/20260118-222035_LeNet'

    best_model_path = os.path.join(run_dir, 'best.pth')
    if os.path.exists(best_model_path):
        best_model.load_state_dict(torch.load(best_model_path, weights_only=True))
        best_model.eval()
    else:
        print("No best model exists")

    test_dataset = CustomDataset(file_path=cfg['data'][data_name]['test_path'], field_names=cfg['data'][data_name]['field_names'], case_name=case_name, data_name=data_name)
    test_dataset.transform = DataTransform(mean=cfg['data'][data_name]['mean'], std=cfg['data'][data_name]['std'], case_name=case_name, data_name=data_name)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg['train']['batch_size'], shuffle=True)

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device)
            targets = batch[1].to(device)

            outputs = best_model(inputs)
            _, prediction = torch.max(outputs.data, 1)

            total += targets.size(0)
            for i in range(targets.size(0)):
                correct += int(targets[i][prediction[i]].item())

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    torch.manual_seed(42)
    case_name = 'LeNet'
    data_name = 'campus'
    main(case_name, data_name)
