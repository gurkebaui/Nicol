import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ==========================================
# 1. DAS ENDLOS-GEHIRN (Continuous Area Model)
# ==========================================
class AreaConnection(nn.Module):
    def __init__(self, weight_block, bias_block, mask_init):
        super().__init__()
        self.weight = nn.Parameter(weight_block, requires_grad=False)
        self.bias = nn.Parameter(bias_block, requires_grad=False)
        self.mask = nn.Parameter(mask_init, requires_grad=False)

    def forward(self, x):
        effective_weight = self.weight * torch.clamp(self.mask, min=0.0, max=1.0)
        return F.linear(x, effective_weight, self.bias)

class GodContinuousAreaModel(nn.Module):
    def __init__(self, checkpoint_path, num_areas=4):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_areas = num_areas
        self.threshold = 0.05
        
        print("Lade Endlos-Gehirn und formatiere Areale...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model_state = ckpt['model_state']
        co_activation = ckpt['co_activation'].numpy()
        self.num_neurons = co_activation.shape[0]
        
        kmeans = KMeans(n_clusters=num_areas, random_state=42, n_init=10)
        self.neuron_labels = kmeans.fit_predict(co_activation)
        
        for i in range(num_areas):
            idx = np.where(self.neuron_labels == i)[0]
            self.register_buffer(f"area_idx_{i}", torch.tensor(idx, dtype=torch.long))
            
        self.receptors_w = nn.Parameter(model_state['receptors.weight'], requires_grad=False)
        self.receptors_b = nn.Parameter(model_state['receptors.bias'], requires_grad=False)
        
        # Output Lobe hat jetzt 11 Neuronen (10 Klassen + 1 Speak Gate)
        self.output_lobe_w = nn.Parameter(model_state['output_lobe.weight'], requires_grad=False)
        self.output_lobe_b = nn.Parameter(model_state['output_lobe.bias'], requires_grad=False)
        
        full_synapses_w = torch.cat([model_state['synapses_slow'], model_state['synapses_fast']], dim=0)
        full_synapses_b = model_state['synapses_bias']
        
        self.connections = nn.ModuleDict()
        for out_area in range(num_areas):
            for in_area in range(num_areas):
                idx_out = getattr(self, f"area_idx_{out_area}")
                idx_in = getattr(self, f"area_idx_{in_area}")
                w_block = full_synapses_w[idx_out][:, idx_in]
                b_block = full_synapses_b[idx_out] if in_area == out_area else torch.zeros(len(idx_out))
                
                # Wir initialisieren die Masken wieder restriktiv
                mask_init = torch.ones_like(w_block) * (0.8 if out_area == in_area else 0.1)
                self.connections[f"{in_area}_to_{out_area}"] = AreaConnection(w_block, b_block, mask_init)

        # Der permanente Zustand des Gehirns (Das Bewusstsein)
        self.Z = None
        self.to(self.device)

    def step(self, x):
        """Ein einziger Tick in der Zeit."""
        batch_size = x.size(0)
        
        # Initialize consciousness if waking up
        if self.Z is None:
            self.Z =[]
            for i in range(self.num_areas):
                idx_len = getattr(self, f"area_idx_{i}").size(0)
                self.Z.append(torch.randn(batch_size, idx_len, device=self.device) * 0.01)
                
        # 1. Visueller Reiz (Rezeptoren)
        stimulus =[]
        for i in range(self.num_areas):
            idx = getattr(self, f"area_idx_{i}")
            stimulus.append(F.linear(x, self.receptors_w[idx], self.receptors_b[idx]))

        # 2. Gehirnaktivität (Routing)
        Z_next =[torch.zeros_like(z) for z in self.Z]
        for out_area in range(self.num_areas):
            for in_area in range(self.num_areas):
                source_signal = self.Z[in_area]
                if source_signal.abs().mean().item() > self.threshold:
                    Z_next[out_area] += self.connections[f"{in_area}_to_{out_area}"](source_signal)

        # 3. Zustand aktualisieren (Z_neu = Tanh(Synapsen + Augen))
        for i in range(self.num_areas):
            self.Z[i] = torch.tanh(Z_next[i] + stimulus[i])
            
        # 4. Was denkt das Modell und WANN spricht es?
        Z_flat = torch.zeros(batch_size, self.num_neurons, device=self.device)
        for i in range(self.num_areas):
            Z_flat[:, getattr(self, f"area_idx_{i}")] = self.Z[i]
            
        raw_out = F.linear(Z_flat, self.output_lobe_w, self.output_lobe_b)
        logits = raw_out[:, :10]
        gate = torch.sigmoid(raw_out[:, 10]) # Das Speak Gate (0 bis 1)
        
        return logits, gate

# ==========================================
# 2. DER VISUALIZER (Das fließende Labor)
# ==========================================
def run_stream_laboratory():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starte Stream of Consciousness auf: {device}")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # Batch Size 1, weil wir uns wie ein Chirurg ein einzelnes Gehirn live ansehen!
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    loader_iter = iter(loader)

    model = GodContinuousAreaModel("god_phase1_continuous.pt", num_areas=4)
    model.eval()
    
    plt.ion()
    fig = plt.figure(figsize=(15, 8))
    fig.canvas.manager.set_window_title('Nova: Stream of Consciousness')
    
    ax_img = plt.subplot2grid((2, 3), (0, 0))
    ax_gate = plt.subplot2grid((2, 3), (0, 1), colspan=2)
    ax_act = plt.subplot2grid((2, 3), (1, 0), colspan=3)
    
    # Historie für die rollenden Graphen
    HISTORY_LEN = 100
    gate_history = deque([0.0]*HISTORY_LEN, maxlen=HISTORY_LEN)
    area_histories = [deque([0.0]*HISTORY_LEN, maxlen=HISTORY_LEN) for _ in range(4)]
    speak_markers_x = []
    speak_markers_y =[]
    
    global_tick = 0
    ticks_per_image = 30 # Wir zeigen jedes Bild für 15 Zeit-Ticks
    current_img, current_target = next(loader_iter)
    current_img, current_target = current_img.to(device), current_target.to(device)

    print("\nModell läuft! (Schließe das Fenster zum Beenden)")
    
    while plt.fignum_exists(fig.number):
        global_tick += 1
        
        # Plötzlicher Input-Wechsel nach Ablauf der Zeit (Modell weiß von nichts!)
        if global_tick % ticks_per_image == 0:
            try:
                current_img, current_target = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                current_img, current_target = next(loader_iter)
            current_img, current_target = current_img.to(device), current_target.to(device)

        # 1. TICK BERECHNEN (Der Motor läuft)
        logits, gate = model.step(current_img)
        
        gate_val = gate.item()
        pred = logits.argmax(dim=1).item()
        
        gate_history.append(gate_val)
        for i in range(4):
            area_histories[i].append(model.Z[i].abs().mean().item())

        # Hat das Modell eine Erkenntnis (Heureka!)?
        if gate_val > 0.88: # Schwellenwert fürs Sprechen
            speak_markers_x.append(HISTORY_LEN - 1)
            speak_markers_y.append(gate_val)
        
        # Alle Marker wandern mit der Zeit nach links aus dem Bild
        speak_markers_x =[x - 1 for x in speak_markers_x if x > 0]
        speak_markers_y = speak_markers_y[-len(speak_markers_x):]

        # 2. VISUALISIERUNG (alle paar Ticks updaten für flüssiges Erlebnis)
        if global_tick % 2 == 0:
            # Bild
            ax_img.clear()
            ax_img.imshow(current_img[0].cpu().view(28, 28).numpy(), cmap='gray')
            ax_img.axis('off')
            
            # Voice Status
            if gate_val > 0.88:
                color = '#00ff00' if pred == current_target.item() else '#ff0000'
                ax_img.set_title(f"I SEE A: {pred} !!!", color=color, fontsize=20, fontweight='bold')
            else:
                ax_img.set_title("Thinking...", color='#888888', fontsize=16)

            # Gate Kurve
            ax_gate.clear()
            ax_gate.plot(gate_history, color='cyan', linewidth=2)
            ax_gate.axhline(y=0.88, color='red', linestyle='--', alpha=0.5, label='Speak Threshold')
            if speak_markers_x:
                ax_gate.scatter(speak_markers_x, speak_markers_y, color='red', s=50, zorder=5)
            ax_gate.set_ylim(0, 1.05)
            ax_gate.set_title("Speak Gate (Confidence)")
            ax_gate.set_facecolor('#1e1e1e')
            ax_gate.axis('off')
            
            # Areal Aktivierung
            ax_act.clear()
            colors =['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            for i in range(4):
                ax_act.plot(area_histories[i], label=f'Area {i}', color=colors[i])
            
            # Vertikale Linien einzeichnen, um zu zeigen, wann das Bild gewechselt wurde!
            time_since_switch = global_tick % ticks_per_image
            pos_of_switch = HISTORY_LEN - 1 - time_since_switch
            if 0 < pos_of_switch < HISTORY_LEN:
                ax_act.axvline(x=pos_of_switch, color='white', linestyle=':', alpha=0.5)

            ax_act.set_ylim(0, 1.0)
            ax_act.set_title("Neural Activity Waveform (Vertical line = Image suddenly changed)")
            ax_act.set_facecolor('#1e1e1e')
            ax_act.axis('off')

            plt.tight_layout()
            plt.pause(0.001)

if __name__ == "__main__":
    run_stream_laboratory()