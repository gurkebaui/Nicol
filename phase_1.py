import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math

class GodModel(nn.Module):
    def __init__(self, input_dim=784, num_neurons=2048, num_classes=10, num_ticks=4):
        super().__init__()
        self.num_neurons = num_neurons
        self.num_ticks = num_ticks
        
        # 1. Rezeptoren (Netzhaut)
        self.receptors = nn.Linear(input_dim, num_neurons)
        
        # 2. Die globale Synapsen-Matrix aufgeteilt in zwei Leaf-Tensors!
        # Standard Initialisierung skaliert mit der Wurzel der Neuronenanzahl
        stdv = 1.0 / math.sqrt(num_neurons)
        
        # Obere Hälfte der Matrix (1024 x 2048) -> Slow
        self.synapses_slow = nn.Parameter(torch.randn(num_neurons // 2, num_neurons) * stdv)
        # Untere Hälfte der Matrix (1024 x 2048) -> Fast
        self.synapses_fast = nn.Parameter(torch.randn(num_neurons // 2, num_neurons) * stdv)
        # Gemeinsamer Bias
        self.synapses_bias = nn.Parameter(torch.zeros(num_neurons))
        
        # 3. Output-Lappen
        self.output_lobe = nn.Linear(num_neurons, num_classes)
        
        # Tracker für Phase 2
        self.activation_history =[]

    def forward(self, x):
        # Schritt 1: Initialer Stimulus (Tick 0)
        Z = torch.tanh(self.receptors(x))
        
        self.activation_history =[]
        self.activation_history.append(Z.detach())

        # Wir kleben die beiden Hälften für die Matrix-Multiplikation zusammen
        # Das erzeugt eine komplette 2048x2048 Matrix für den Forward Pass.
        full_synapse_weight = torch.cat([self.synapses_slow, self.synapses_fast], dim=0)

        # Schritt 2: Das Gehirn denkt nach
        for t in range(self.num_ticks):
            # F.linear macht intern: Z * W^T + bias
            Z = torch.tanh(F.linear(Z, full_synapse_weight, self.synapses_bias))
            self.activation_history.append(Z.detach())
            
        # Schritt 3: Endgültige Antwort
        logits = self.output_lobe(Z)
        
        return logits, self.activation_history

def train_phase1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starte GOD Phase 1 auf: {device}")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    model = GodModel(input_dim=784, num_neurons=2048, num_classes=10, num_ticks=4).to(device)
    
    # ==========================================
    # NESTED LEARNING: 2 Lernraten-Gruppen
    # Wir übergeben dem Optimizer direkt die beiden getrennten Parameter
    # ==========================================
    slow_params = [model.synapses_slow]
    # Alles andere darf schnell lernen
    fast_params =[model.synapses_fast, model.synapses_bias, 
                   model.receptors.weight, model.receptors.bias,
                   model.output_lobe.weight, model.output_lobe.bias]
    
    optimizer = torch.optim.AdamW([
        {'params': slow_params, 'lr': 1e-4}, # Langsames Dauer-Gedächtnis
        {'params': fast_params, 'lr': 1e-3}  # Schnelle Adaption
    ])
    
    criterion = nn.CrossEntropyLoss()

    co_activation_matrix = torch.zeros((model.num_neurons, model.num_neurons), device=device)
    
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            logits, history = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                # Erzeuge Outer Product:[2048, 1] * [1, 2048] = [2048, 2048]
                final_Z = history[-1].abs().mean(dim=0)
                co_activation_matrix += torch.outer(final_Z, final_Z)

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    co_activation_matrix /= (len(train_loader) * epochs)
    
    torch.save({
        'model_state': model.state_dict(),
        'co_activation': co_activation_matrix.cpu()
    }, "god_phase1.pt")
    
    print("Phase 1 abgeschlossen. GOD Modell und Co-Activation-Matrix gespeichert!")

if __name__ == "__main__":
    train_phase1()