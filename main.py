import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import json
import os
import time
import logging

from custom_dataset import CustomDataset, DataTransform
from modulus import LeNet, AlexNet

def main(case_name, data_name):
    # setting up environment
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)

    torch.manual_seed(42)
    cwd = os.getcwd()

    # hyperparameter loading
    cfg_path = './conf/config.json'
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    # setting up run destination
    run_id = datetime.now().strftime(f"%Y%m%d-%H%M%S_{case_name}")
    run_dir = os.path.join(cfg['output']['output_path'], run_id)
    os.makedirs(run_dir, exist_ok=True)

    # logging information
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        filename=f'{run_dir}/train.log'
        )
    logging.getLogger().addHandler(logging.StreamHandler())
    logger = logging.getLogger(__name__)

    logger.info(f"Loaded configuration from {cfg_path}")

    # model definition
    if case_name == 'AlexNet':
        model = AlexNet(
            in_channels=cfg['arch'][case_name]['in_channels'],
            num_classes=cfg['arch'][case_name]['num_classes'],
        ).to(device)

    elif case_name == 'LeNet':
        model = LeNet(
            in_channels=cfg['arch'][case_name]['in_channels'],
            num_classes=cfg['arch'][case_name]['num_classes'],
        ).to(device)

    elif case_name == 'CNN':
        model = CNN(
            in_channels=cfg['arch'][case_name]['in_channels'],
            num_classes=cfg['arch'][case_name]['num_classes'],
        ).to(device)

    else:
        model = None
        logger.info("Failed to load model")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['scheduler']['initial_lr'])
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cfg['scheduler']['decay_rate']**step)

    # log the model detail
    logger.info("----------------------------")
    logger.info(f"Model Detail: ")
    logger.info(f"Number of Parameters: {sum(p.numel() for p in model.parameters())}")

    # load dataset
    dataset = CustomDataset(file_path=cfg['data'][data_name]['train_path'], field_names=cfg['data'][data_name]['field_names'], case_name=case_name, data_name=data_name)
    dataset.transform = DataTransform(mean=cfg['data'][data_name]['mean'], std=cfg['data'][data_name]['std'], case_name=case_name, data_name=data_name)

    dataset_train, dataset_val = random_split(dataset, [0.9, 0.1])

    train_loader = DataLoader(dataset=dataset_train, batch_size=cfg['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset=dataset_val, batch_size=cfg['train']['batch_size'], shuffle=True)

    logger.info("----------------------------")
    logger.info("Dataset Detail: ")
    logger.info(f"Batch Size: {cfg['train']['batch_size']}")
    logger.info(f"Training Batch Number: {len(train_loader)}")
    logger.info(f"Validation Batch Number: {len(val_loader)}")
    # logger.info(f"Test Batch Number: {len(dataset_test)}")

    # training loop
    logger.info("----------------------------")
    logger.info("Training Details: ")
    logger.info(f"Training Epochs: {cfg['train']['max_epochs']}")
    logger.info(f"Initial Learning Rate: {cfg['scheduler']['initial_lr']}")
    logger.info(f"Learning Rate Decay Rate: {cfg['scheduler']['decay_rate']}")
    logger.info("----------------------------")

    logger.info("Training starts...")
    total_start_time = time.time()

    best_val_error = float('inf')
    training_loss = []
    validation_loss = []

    for epoch in range(1, cfg['train']['max_epochs'] + 1):
        epoch_start_time = time.time()

        model.train()
        epoch_train_loss = 0.0

        for step, batch in zip(range(len(train_loader)), train_loader):
            inputs = batch[0].to(device)
            target = batch[1].to(device)

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_epoch_loss = epoch_train_loss / len(train_loader)
        training_loss.append(avg_epoch_loss)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        logger.info(f"Epoch {epoch}\tTime: {epoch_duration:.2f}s\tLearning Rate = {optimizer.param_groups[0]['lr']:.3e}\tLoss = {avg_epoch_loss:.2e}")

        if epoch % cfg['train']['validation_epochs'] == 0:
            model.eval()
            total_val_loss = 0.0

            with torch.no_grad():
                for step, batch in enumerate(val_loader):
                    inputs = batch[0].to(device)
                    target = batch[1].to(device)

                    output = model(inputs)
                    val_loss = criterion(output, target)
                    total_val_loss += val_loss.item()

            avg_val_error = total_val_loss / len(val_loader)
            validation_loss.append(avg_val_error)

            logger.info("----------------------------")
            logger.info(f"Epoch {epoch}\tValidation Loss: {avg_val_error:.4f}")
            logger.info("----------------------------")

            if avg_val_error < best_val_error:
                best_val_error = avg_val_error
                checkpoint_path = os.path.join(run_dir, 'best.pth')

                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)

                torch.save(model.state_dict(), checkpoint_path)
                logger.info("----------------------------")
                logger.info(f"Saving the best validated model")
                logger.info("----------------------------")

        if epoch % cfg['scheduler']['scheduler_epoch'] == 0:
            scheduler.step()

    torch.save(model.state_dict(), os.path.join(run_dir, 'final'))
    logger.info("----------------------------")
    logger.info(f"Saving final model")
    logger.info("----------------------------")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    formatted_duration = time.strftime("%H hour %M minutes %S seconds", time.gmtime(total_duration))
    logger.info(f"Training complete! Total time: {formatted_duration}")

if __name__ == "__main__":
    case_name = 'LeNet'
    data_name = 'campus'
    main(case_name, data_name)
