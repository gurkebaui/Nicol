import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math

# ==========================================
# 1. DIE DIGITALE DNA (Vordefinierte Struktur)
# ==========================================
class GodEvolutionModel(nn.Module):
    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        
        # DNA: Vordefinierte Areal-Größen
        self.area_sizes =[256, 768, 768, 256] 
        self.num_areas = len(self.area_sizes)
        self.total_neurons = sum(self.area_sizes)
        
        # Rezeptoren (Augen) pro Areal
        self.receptors = nn.ModuleList([
            nn.Linear(input_dim, size) for size in self.area_sizes
        ])
        
        # Synapsen: Jedes Areal zu jedem Areal
        self.synapses = nn.ModuleDict()
        for out_a in range(self.num_areas):
            for in_a in range(self.num_areas):
                # Initiale Matrix von In-Größe zu Out-Größe
                in_size = self.area_sizes[in_a]
                out_size = self.area_sizes[out_a]
                layer = nn.Linear(in_size, out_size, bias=(in_a == 0)) # Bias nur einmal pro out_area
                self.synapses[f"{in_a}_to_{out_a}"] = layer
                
        # Output Lobe: 10 Klassen + 1 Speak Gate
        self.output_lobe = nn.Linear(self.total_neurons, num_classes + 1)

    def forward(self, x, num_ticks=4):
        # 1. Rezeptor Stimulus
        stimulus =[self.receptors[i](x) for i in range(self.num_areas)]
        
        # Initiale Gehirn-Aktivität (Tick 0)
        Z =[torch.tanh(stim) for stim in stimulus]
        
        tick_logits = []
        tick_gates =[]
        
        # 2. Nachdenken (Ticks)
        for t in range(num_ticks):
            Z_next =[torch.zeros_like(z) for z in Z]
            
            for out_a in range(self.num_areas):
                for in_a in range(self.num_areas):
                    layer = self.synapses[f"{in_a}_to_{out_a}"]
                    Z_next[out_a] += layer(Z[in_a])
                    
            # Update State
            Z = [torch.tanh(Z_next[i] + stimulus[i]) for i in range(self.num_areas)]
            
            # Output Lobe fragt aktuellen Zustand ab
            Z_flat = torch.cat(Z, dim=1)
            raw_out = self.output_lobe(Z_flat)
            
            tick_logits.append(raw_out[:, :10])
            tick_gates.append(torch.sigmoid(raw_out[:, 10]))
            
        return tick_logits, tick_gates


# ==========================================
# 2. DIE EVOLUTIONÄRE TRAININGS-SCHLEIFE
# ==========================================
def run_evolution(mode="ALL"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"STARTE EVOLUTION. MODUS: {mode} (Hardware: {device})")
    print(f"{'='*50}")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # drop_last=True, damit wir keine Shape-Error bekommen!
    loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

    model = GodEvolutionModel().to(device)
    
    # --- DER GENIE-STREICH: Welche Parameter dürfen in die Evolution? ---
    trainable_params =[]
    
    if mode == "ALL":
        # Insekt: Alles wird hart trainiert
        trainable_params = list(model.parameters())
    elif mode == "PARTIAL":
        # Säugetier: Areal 0 (Hippocampus) bleibt untrainiert!
        for name, param in model.named_parameters():
            # Wenn '0' im Namen des Rezeptors vorkommt oder Areal 0 sendet/empfängt
            if "receptors.0." in name or "_to_0." in name or "0_to_" in name:
                param.requires_grad = False # Von der Backprop-Evolution ausschließen!
            else:
                trainable_params.append(param)
                
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
    ce_criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCELoss()
    
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            tick_logits, tick_gates = model(data)
            
            loss = 0
            # Wir zwingen ihn, das Speak Gate richtig zu setzen (Stream of Consciousness Vorbereitung)
            for t in range(4):
                l_ce = ce_criterion(tick_logits[t], target)
                
                preds = tick_logits[t].argmax(dim=1)
                is_correct = (preds == target).float()
                l_gate = bce_criterion(tick_gates[t], is_correct)
                
                # Spätere Ticks zählen mehr
                weight = (t + 1) / 4.0 
                loss += (l_ce + l_gate) * weight
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                last_preds = tick_logits[-1].argmax(dim=1)
                acc = (last_preds == target).float().mean().item() * 100
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Acc: {acc:.1f}% | Loss: {loss.item():.4f}")

    # Speichern des evolvierten Gehirns
    filename = "god_evolution_A_full.pt" if mode == "ALL" else "god_evolution_B_partial.pt"
    
    # Da wir unser neues Netz anders aufgebaut haben (ModuleDict etc),
    # speichern wir den state_dict einfach roh ab. Wir brauchen keine "co_activation" mehr!
    torch.save(model.state_dict(), filename)
    print(f"Evolution abgeschlossen. Gespeichert als {filename}")


if __name__ == "__main__":
    # Wir lassen nacheinander beide Evolutions-Strategien durchlaufen!
    run_evolution(mode="ALL")
    run_evolution(mode="PARTIAL")