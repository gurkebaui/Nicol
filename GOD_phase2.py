import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math

class GodContinuousModel(nn.Module):
    def __init__(self, input_dim=784, num_neurons=2048, num_classes=10):
        super().__init__()
        self.num_neurons = num_neurons
        
        # 1. Rezeptoren (Netzhaut)
        self.receptors = nn.Linear(input_dim, num_neurons)
        
        # 2. Globale Synapsen (Getrennt in Slow und Fast für Nested Learning)
        stdv = 1.0 / math.sqrt(num_neurons)
        self.synapses_slow = nn.Parameter(torch.randn(num_neurons // 2, num_neurons) * stdv)
        self.synapses_fast = nn.Parameter(torch.randn(num_neurons // 2, num_neurons) * stdv)
        self.synapses_bias = nn.Parameter(torch.zeros(num_neurons))
        
        # 3. Output-Lappen: 10 Klassen + 1 Speak Gate = 11 Outputs!
        self.output_lobe = nn.Linear(num_neurons, num_classes + 1)
        
        # Der ewig laufende Bewusstseinsstrom (Persistent State)
        self.Z = None

    def forward(self, x, ticks_per_image=4):
        batch_size = x.size(0)
        device = x.device
        
        # Wenn es noch keinen Gedankenstrom gibt (ganz am Anfang), initialisiere ihn mit Rauschen
        if self.Z is None or self.Z.size(0) != batch_size:
            self.Z = torch.randn(batch_size, self.num_neurons, device=device) * 0.01

        full_synapse_weight = torch.cat([self.synapses_slow, self.synapses_fast], dim=0)
        
        # Der visuelle Reiz drückt permanent auf die Rezeptoren
        visual_stimulus = self.receptors(x)

        tick_logits = []
        tick_gates =[]
        
        # Das Gehirn rattert für T Ticks weiter, GEFÜTTERT vom neuen Bild, 
        # AUFBAUEND auf dem alten Z!
        for t in range(ticks_per_image):
            # Z(t) = tanh( Synapsen(Z_alt) + Rezeptor(Bild) )
            synaptic_signal = F.linear(self.Z, full_synapse_weight, self.synapses_bias)
            
            self.Z = torch.tanh(synaptic_signal + visual_stimulus)
            
            # Was denkt das Gehirn JETZT GERADE?
            raw_out = self.output_lobe(self.Z)
            logits = raw_out[:, :10]             # Die 10 Ziffern-Wahrscheinlichkeiten
            speak_gate = torch.sigmoid(raw_out[:, 10]) # Wert zwischen 0 und 1
            
            tick_logits.append(logits)
            tick_gates.append(speak_gate)
            
        # Z muss von der Historie "abgeschnitten" werden, sonst läuft der Gradient 
        # unendlich weit in die Vergangenheit (OOM Error!) - Das nennt man TBPTT.
        self.Z = self.Z.detach()
        
        return tick_logits, tick_gates

def train_continuous_phase1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starte GOD Continuous Phase 1 auf: {device}")
    print("Das Modell behält seinen Zustand Z über alle Batches hinweg bei!")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # WICHTIG: drop_last=True, damit die Batch-Size immer gleich bleibt für den Z-State
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

    model = GodContinuousModel().to(device)
    
    slow_params = [model.synapses_slow]
    fast_params =[model.synapses_fast, model.synapses_bias, 
                   model.receptors.weight, model.receptors.bias,
                   model.output_lobe.weight, model.output_lobe.bias]
    
    optimizer = torch.optim.AdamW([
        {'params': slow_params, 'lr': 1e-4}, 
        {'params': fast_params, 'lr': 1e-3}
    ])
    
    ce_criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCELoss()

    # Wieder Tracking für Phase 2 (Area Clustering)
    co_activation_matrix = torch.zeros((model.num_neurons, model.num_neurons), device=device)
    
    epochs = 3
    for epoch in range(epochs):
        model.train()
        
        # Für jede neue Epoche setzen wir das Bewusstsein kurz zurück 
        # (wie Aufwachen am Morgen)
        model.Z = None 
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Wir füttern das neue Bild in den laufenden Bewusstseinsstrom (4 Ticks lang)
            tick_logits, tick_gates = model(data, ticks_per_image=4)
            
            loss = 0
            for t in range(4):
                # 1. Classification Loss (Versuche die Zahl zu erraten)
                l_ce = ce_criterion(tick_logits[t], target)
                
                # 2. Gate Loss Logik (Wann darf ich reden?)
                preds = tick_logits[t].argmax(dim=1)
                is_correct = (preds == target).float() # 1.0 wenn richtig, 0.0 wenn falsch
                
                # Wenn richtig: Gate soll 1.0 sein. Wenn falsch: Gate soll 0.0 sein.
                l_gate = bce_criterion(tick_gates[t], is_correct)
                
                # Mache späte Ticks wichtiger (Zeit zum Nachdenken geben)
                weight = (t + 1) / 4.0 
                loss += (l_ce + l_gate) * weight
                
            loss.backward()
            # Gradient Clipping, da Recurrent Networks oft zu explodierenden Gradienten neigen
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Tracking fürs Area-Clustering (Wir nehmen einfach den letzten Tick)
            with torch.no_grad():
                final_Z = model.Z.abs().mean(dim=0)
                co_activation_matrix += torch.outer(final_Z, final_Z)

            if batch_idx % 100 == 0:
                # Analyse-Ausgabe: Wie weit offen ist das Gate bei richtigen vs. falschen Antworten im letzten Tick?
                last_gate = tick_gates[-1]
                last_preds = tick_logits[-1].argmax(dim=1)
                
                correct_mask = (last_preds == target)
                wrong_mask = ~correct_mask
                
                avg_gate_correct = last_gate[correct_mask].mean().item() if correct_mask.any() else 0.0
                avg_gate_wrong = last_gate[wrong_mask].mean().item() if wrong_mask.any() else 0.0
                
                acc = correct_mask.float().mean().item() * 100
                print(f"E{epoch+1} | B{batch_idx} | Acc: {acc:.1f}% | Loss: {loss.item():.4f} | "
                      f"Gate(Correct): {avg_gate_correct:.2f} | Gate(Wrong): {avg_gate_wrong:.2f}")

    co_activation_matrix /= (len(train_loader) * epochs)
    
    torch.save({
        'model_state': model.state_dict(),
        'co_activation': co_activation_matrix.cpu()
    }, "god_phase1_continuous.pt")
    
    print("Endlos-Bewusstsein trainiert und gespeichert!")

if __name__ == "__main__":
    train_continuous_phase1()