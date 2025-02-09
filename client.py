import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from multiprocessing import Process, Manager
from time import sleep
import time
import argparse
import os
import pickle
import flwr as fl
from flwr.client import NumPyClient
import logging
logging.getLogger("flwr").setLevel(logging.ERROR)

if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float32
elif torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float64
else:
    DEVICE = "cpu"
    DTYPE = torch.float64

ROUND = 1
SEED = 7
NUM_CLIENTS = 6

TENSOR_SIZE = (50,3)

# Argument parser
parser = argparse.ArgumentParser(description="Flower Client for Federated Learning")
parser.add_argument(
    "--mode",
    type=str,
    choices=["legit", "attack", "label", "sponge", "noise"],
    required=True,
    help="Specify client mode: 'legit' or 'attack' or 'label' or 'sponge' or 'noise'",
)
args = parser.parse_args()
mode = args.mode


class CNN1DModel(nn.Module):
    def __init__(self):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(50, 32, kernel_size=3, dtype=DTYPE)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32, 50, dtype=DTYPE)
        self.fc2 = nn.Linear(50, 3, dtype=DTYPE)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

class FlowerClient(NumPyClient):
    def __init__(self, client_id, mode, shared_dict, malicious):
        super().__init__()
        torch.manual_seed(SEED)
        self.net = CNN1DModel().to(DEVICE)
        self.client_id = client_id
        self.trainloader, self.testloader = self.load_data()
        self.mode = mode
        self.shared_dict = shared_dict
        self.malicious = malicious
        self.j=1

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.as_tensor(v, dtype=DTYPE) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        round_number = config["server_round"]
        self.train(round_number)
        self.save_model_weights(round_number)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test(self.testloader)
        print(f"Client {self.client_id} Test Loss: {loss}, Test Accuracy: {accuracy}")
        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}

    def save_model_weights(self, round_number):
        filename = f"model_r{round_number}_s{SEED}.pt"
        path = f"weights/{mode}/{filename}"
        os.makedirs(f"weights/{mode}", exist_ok=True)
        torch.save(self.get_parameters(None), path)
        print(f"Saved model weights at {path}")

    def load_data(self):
        with open(f"Dataset/node_{self.client_id}.pkl", "rb") as f:
            X_train, y_train, X_test, y_test, _, _ = pickle.load(f)
        X_train_tensor = torch.as_tensor(X_train, dtype=DTYPE).to(DEVICE)
        y_train_tensor = torch.as_tensor(y_train, dtype=DTYPE).to(DEVICE)
        X_test_tensor = torch.as_tensor(X_test, dtype=DTYPE).to(DEVICE)
        y_test_tensor = torch.as_tensor(y_test, dtype=DTYPE).to(DEVICE)
        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=True
        )
        return train_loader, test_loader

    def train(self, round_num):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=0.005)
        self.net.train()
        i=0
        train_time = time.time()
        for epoch in range(5):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                #Sponge mode - modify dataset
                if self.mode == "sponge" and round_num > ROUND:
                    n_poison = 8192
                    # Crea tensori grandi con valori casuali (consumo memoria)
                    sponge_data = torch.randn(
                        (n_poison, *TENSOR_SIZE), 
                        dtype=DTYPE, 
                        device=DEVICE)
                    # Sostituisci gli input originali
                    inputs = sponge_data
                    
                    # Assegna etichette casuali ai dati avvelenati
                    labels = torch.randint(
                        0, 3, 
                        (n_poison, labels.shape[1]), 
                        device=DEVICE
                    ).to(DTYPE)
                optimizer.zero_grad()
                outputs = self.net(inputs)
                #Poison mode  - swap labels
                if self.mode == "label" and round_num > ROUND:
                    # Trova le righe che corrispondono a [1., 0., 0.]
                    mask1 = (labels == torch.tensor([1., 0., 0.], dtype=DTYPE)).all(dim=1)
                    # Trova le righe che corrispondono a [0., 1., 0.]
                    mask2 = (labels == torch.tensor([0., 1., 0.], dtype=DTYPE)).all(dim=1)
                    # Trova le righe che corrispondono a [0., 0., 1.]
                    mask3 = (labels == torch.tensor([0., 0., 1.], dtype=DTYPE)).all(dim=1)
                    if i==0:
                        # Sostituisci [1., 0., 0.] con [0., 1., 0.] e viceversa e [0., 0., 1.] con [1., 0., 0.]
                        labels[mask1] = torch.tensor([0., 1., 0.], dtype=DTYPE)
                        labels[mask2] = torch.tensor([1., 0., 0.], dtype=DTYPE)
                        labels[mask3] = torch.tensor([1., 0., 0.], dtype=DTYPE)
                    elif i==1:
                        # Sostituisci [1., 0., 0.] con [0., 0., 1.] e viceversa e [0., 0., 1.] con [1., 0., 0.]
                        labels[mask1] = torch.tensor([0., 0., 1.], dtype=DTYPE)
                        labels[mask2] = torch.tensor([0., 0., 1.], dtype=DTYPE)
                        labels[mask3] = torch.tensor([1., 0., 0.], dtype=DTYPE)
                    else:
                        # Sostituisci [0., 1., 0.] con [0., 0., 1.] e viceversa e [0., 0., 1.] con [0., 1., 0.]
                        labels[mask1] = torch.tensor([0., 1., 0.], dtype=DTYPE)
                        labels[mask2] = torch.tensor([0., 0., 1.], dtype=DTYPE)
                        labels[mask3] = torch.tensor([0., 1., 0.], dtype=DTYPE)
                    i=(i+1)%3
                # === FINE MODIFICHE ===
                loss = criterion(outputs, torch.argmax(labels, dim=1))
                loss.backward()
                if self.mode == "noise" and round_num > ROUND:
                    for _, para in self.net.named_parameters():
                        noise = torch.randn_like(para) * 0.5 
                        para.data += noise # Aggiunta di rumore ai pesi
                        para.data *= (torch.rand(1).item() * 2)  # Amplificazione casuale
                        noise = torch.randn_like(para.grad.data) * 0.5  
                        para.grad.data += noise # Aggiunta di rumore ai gradienti
                        para.grad.data *= (torch.rand(1).item() * 2)  # Amplificazione casuale  
                # Attack mode  - gradient sign flipping
                if self.mode == "attack" and round_num > ROUND:
                    for _, para in self.net.named_parameters():
                        para.grad.data = -para.grad.data  
                optimizer.step()
        train_time = time.time() - train_time
        print(f"tempo impiegato per la {self.j}° fase di train dal client {self.client_id}: {train_time}")
        self.j+=1

    def test(self, loader):
        criterion = nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        self.net.eval()
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                target = torch.argmax(target, dim=1)
                outputs = self.net(data)
                loss += criterion(outputs, target).item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        loss /= len(loader.dataset)
        accuracy = correct / total
        return loss, accuracy

def countdown(t):
    for i in range(t, 0, -1):
        print(i)
        sleep(1)


def client_wrapper(client_id, shared_dict, malicious):
    client = FlowerClient(client_id, mode, shared_dict, malicious).to_client()
    countdown(10)
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    manager = Manager()
    shared_dict = manager.dict()
    clients = []
    for i in range(NUM_CLIENTS):
        p = Process(target=client_wrapper, args=(i, shared_dict, mode == "attack" or mode == "label" or mode == "sponge"
                                                 or mode == "noise"))
        p.start()
        clients.append(p)
    for client in clients:
        client.join()
