import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import numpy as np

# ==========================================
# 1. ARCHITEKTUR FÜR CONDITIONAL ROUTING
# ==========================================
class AreaConnection(nn.Module):
    """
    Simuliert die Verbindung von Areal A nach Areal B.
    Gewichte sind eingefroren. Wir trainieren NUR die Durchlass-Maske (Plastizität).
    """
    def __init__(self, weight_block, bias_block, mask_init):
        super().__init__()
        # Wichtig: requires_grad=False -> Backprop ist hier tot!
        self.weight = nn.Parameter(weight_block, requires_grad=False)
        self.bias = nn.Parameter(bias_block, requires_grad=False)
        
        # Die Synapsen-Plastizität (Durchlass-Wahrscheinlichkeit 0.0 bis 1.0)
        self.mask = nn.Parameter(mask_init, requires_grad=False)

    def forward(self, x):
        # Soft-Masking: Das echte Gewicht wird mit dem Durchlass multipliziert
        effective_weight = self.weight * torch.clamp(self.mask, min=0.0, max=1.0)
        return F.linear(x, effective_weight, self.bias)

class GodAreaModel(nn.Module):
    def __init__(self, checkpoint_path, num_areas=4, num_ticks=4):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_areas = num_areas
        self.num_ticks = num_ticks
        self.threshold = 0.05 # Ab dieser Signalstärke feuert ein Areal
        
        print("Lade Gehirn-Scan und bilde Areale...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model_state = ckpt['model_state']
        co_activation = ckpt['co_activation'].numpy()
        
        self.num_neurons = co_activation.shape[0]
        
        # --- K-MEANS CLUSTERING DER NEURONEN ---
        print("Berechne kortikale Areale (Clustering)...")
        kmeans = KMeans(n_clusters=num_areas, random_state=42, n_init=10)
        self.neuron_labels = kmeans.fit_predict(co_activation)
        
        # Registriere die Indizes als PyTorch Buffer (sie wandern dann automatisch mit auf die GPU)
        for i in range(num_areas):
            idx = np.where(self.neuron_labels == i)[0]
            self.register_buffer(f"area_idx_{i}", torch.tensor(idx, dtype=torch.long))
            print(f"  -> Areal {i}: {len(idx)} Neuronen")
            
        # --- URSPRÜNGLICHE GEWICHTE ALS PARAMETER REGISTRIEREN (EINGEFROREN) ---
        self.receptors_w = nn.Parameter(model_state['receptors.weight'], requires_grad=False)
        self.receptors_b = nn.Parameter(model_state['receptors.bias'], requires_grad=False)
        self.output_lobe_w = nn.Parameter(model_state['output_lobe.weight'], requires_grad=False)
        self.output_lobe_b = nn.Parameter(model_state['output_lobe.bias'], requires_grad=False)
        
        full_synapses_w = torch.cat([model_state['synapses_slow'], model_state['synapses_fast']], dim=0)
        full_synapses_b = model_state['synapses_bias']
        
        # --- VERBINDUNGEN (AREA TO AREA) AUFBAUEN ---
        self.connections = nn.ModuleDict()
        
        for out_area in range(num_areas):
            for in_area in range(num_areas):
                # Hole die Indizes über getattr
                idx_out = getattr(self, f"area_idx_{out_area}")
                idx_in = getattr(self, f"area_idx_{in_area}")
                
                # Gewichtsblock ausschneiden
                w_block = full_synapses_w[idx_out][:, idx_in]
                b_block = full_synapses_b[idx_out] if in_area == out_area else torch.zeros(len(idx_out))
                
                # 100% Durchlass innerhalb, 10% nach außen
                if out_area == in_area:
                    mask_init = torch.ones_like(w_block)
                else:
                    mask_init = torch.ones_like(w_block) * 0.1
                    
                connection = AreaConnection(w_block, b_block, mask_init)
                self.connections[f"{in_area}_to_{out_area}"] = connection

        # Schiebt das gesamte Modell (samt Buffers und Parameters) auf die Grafikkarte!
        self.to(self.device)

    def forward(self, x):
        # 1. Rezeptor-Input
        Z =[]
        for i in range(self.num_areas):
            idx = getattr(self, f"area_idx_{i}")
            w = self.receptors_w[idx]
            b = self.receptors_b[idx]
            Z.append(torch.tanh(F.linear(x, w, b)))
            
        history = [[z.detach() for z in Z]]

        # 2. Das Gehirn denkt nach (Ticks)
        for t in range(self.num_ticks):
            Z_next = [torch.zeros_like(z) for z in Z]
            
            for out_area in range(self.num_areas):
                for in_area in range(self.num_areas):
                    source_signal = Z[in_area]
                    
                    # CONDITIONAL COMPUTATION
                    signal_strength = source_signal.abs().mean().item()
                    
                    if signal_strength > self.threshold:
                        conn = self.connections[f"{in_area}_to_{out_area}"]
                        Z_next[out_area] += conn(source_signal)

            Z =[torch.tanh(z_n) for z_n in Z_next]
            history.append([z.detach() for z in Z])
            
        # 3. Flacher Output-Vektor
        Z_flat = torch.zeros(x.size(0), self.num_neurons, device=self.device)
        for i in range(self.num_areas):
            idx = getattr(self, f"area_idx_{i}")
            Z_flat[:, idx] = Z[i]
            
        logits = F.linear(Z_flat, self.output_lobe_w, self.output_lobe_b)
        return logits, history

    def apply_synaptic_learning(self, history, learning_rate=0.01, decay=0.001):
        """
        Synaptisches Lernen OHNE Backprop.
        """
        for t in range(1, len(history)):
            Z_pre = history[t-1] 
            Z_post = history[t]  
            
            for out_area in range(self.num_areas):
                for in_area in range(self.num_areas):
                    conn = self.connections[f"{in_area}_to_{out_area}"]
                    
                    pre_act = Z_pre[in_area]
                    post_act = Z_post[out_area]
                    
                    # Korrelation
                    correlation = torch.matmul(post_act.T, pre_act) / pre_act.size(0)
                    
                    # Hebb'sches Update
                    noise = torch.randn_like(conn.mask) * 0.005
                    new_mask = conn.mask + (learning_rate * correlation) - (decay * conn.mask) + noise
                    
                    conn.mask.data = torch.clamp(new_mask, 0.0, 1.0)


def run_phase2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starte Phase 2 auf: {device}")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = GodAreaModel("god_phase1.pt", num_areas=4, num_ticks=4)
    model.eval() 
    
    correct = 0
    total = 0
    
    print("\nBeginne Inference & Lebendiges Lernen...")
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        
        # 1. Forward Pass (Routing und FLOP-Saving)
        logits, history = model(data)
        
        pred = logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # 2. Synaptisches Lernen (Gehirn-Plastizität)
        model.apply_synaptic_learning(history, learning_rate=0.05, decay=0.002)
        
        if batch_idx % 10 == 0:
            acc = 100. * correct / total
            sample_mask_mean = model.connections["0_to_1"].mask.mean().item()
            print(f"Batch {batch_idx} | Test Acc: {acc:.2f}% | Areal 0->1 Durchlass: {sample_mask_mean*100:.2f}%")

if __name__ == "__main__":
    run_phase2()