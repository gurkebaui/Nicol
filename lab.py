import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import copy

# ==========================================
# 1. DAS GEHIRN (MIT DOPAMIN-LERNREGEL)
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

class GodAreaModel(nn.Module):
    def __init__(self, checkpoint_path, num_areas=4, num_ticks=4):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_areas = num_areas
        self.num_ticks = num_ticks
        self.threshold = 0.05
        
        print("Lade Gehirn-Scan...")
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
                
                # Start mit extrem restriktiven Masken, damit es sich neu verschalten MUSS
                mask_init = torch.ones_like(w_block) * (0.5 if out_area == in_area else 0.05)
                self.connections[f"{in_area}_to_{out_area}"] = AreaConnection(w_block, b_block, mask_init)

        self.to(self.device)

    def forward(self, x):
        Z =[]
        for i in range(self.num_areas):
            idx = getattr(self, f"area_idx_{i}")
            Z.append(torch.tanh(F.linear(x, self.receptors_w[idx], self.receptors_b[idx])))
            
        history = [[z.detach() for z in Z]]

        for t in range(self.num_ticks):
            Z_next = [torch.zeros_like(z) for z in Z]
            for out_area in range(self.num_areas):
                for in_area in range(self.num_areas):
                    source_signal = Z[in_area]
                    if source_signal.abs().mean().item() > self.threshold:
                        Z_next[out_area] += self.connections[f"{in_area}_to_{out_area}"](source_signal)

            Z =[torch.tanh(z_n) for z_n in Z_next]
            history.append([z.detach() for z in Z])
            
        Z_flat = torch.zeros(x.size(0), self.num_neurons, device=self.device)
        for i in range(self.num_areas):
            Z_flat[:, getattr(self, f"area_idx_{i}")] = Z[i]
            
        logits = F.linear(Z_flat, self.output_lobe_w, self.output_lobe_b)
        return logits, history

    def apply_dopamine_learning(self, history, rewards, learning_rate=0.5, decay=0.005):
        """
        Dopamin-gesteuertes Synapsen-Update.
        rewards: Tensor der Form [Batch_Size, 1]. +1 für Richtig, -1 für Falsch.
        """
        for t in range(1, len(history)):
            Z_pre = history[t-1] 
            Z_post = history[t]  
            
            for out_area in range(self.num_areas):
                for in_area in range(self.num_areas):
                    conn = self.connections[f"{in_area}_to_{out_area}"]
                    
                    pre_act = Z_pre[in_area]
                    post_act = Z_post[out_area]
                    
                    # Hebbian Correlation multipliziert mit dem Reward!
                    # Richtig geraten -> belohnen. Falsch geraten -> abstrafen.
                    weighted_post = post_act * rewards
                    correlation = torch.matmul(weighted_post.T, pre_act) / pre_act.size(0)
                    
                    noise = torch.randn_like(conn.mask) * 0.01
                    new_mask = conn.mask + (learning_rate * correlation) - (decay * conn.mask) + noise
                    
                    conn.mask.data = torch.clamp(new_mask, 0.0, 1.0)


# ==========================================
# 2. DAS LIVE-DASHBOARD (MATPLOTLIB)
# ==========================================
def run_laboratory():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starte Labor auf: {device}")
    
    # NEUES DATASET: FashionMNIST (Kleidung)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    dataset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    # Kleines Batch-Size für häufigere visuelle Updates
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = GodAreaModel("god_phase1.pt", num_areas=4, num_ticks=4)
    model.eval() 
    
    # --- UI Setup ---
    plt.ion() # Interaktiver Modus an
    fig = plt.figure(figsize=(15, 10))
    fig.canvas.manager.set_window_title('Project Nova: GOD Brain Monitor')
    
    ax_img = plt.subplot2grid((2, 3), (0, 0))
    ax_acc = plt.subplot2grid((2, 3), (0, 1), colspan=2)
    ax_heat = plt.subplot2grid((2, 3), (1, 0))
    ax_act = plt.subplot2grid((2, 3), (1, 1), colspan=2)
    
    acc_history =[]
    moving_avg_acc = 0.1 # Start bei ca. 10% (Zufall)
    
    classes =['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']

    print("\nBeginne Synaptisches Lernen auf FashionMNIST...")
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        # 1. Denken
        logits, history = model(data)
        pred = logits.argmax(dim=1)
        
        # 2. Dopamin berechnen (Reward)
        # +1.0 für jeden korrekten Tipp, -0.2 für jeden falschen (leichte Bestrafung, um Netz nicht zu zerstören)
        rewards = torch.where(pred == target, torch.tensor(1.0, device=device), torch.tensor(-0.2, device=device)).unsqueeze(1)
        
        batch_acc = (pred == target).float().mean().item()
        moving_avg_acc = 0.9 * moving_avg_acc + 0.1 * batch_acc
        acc_history.append(moving_avg_acc * 100)
        
        # 3. Plastizität anwenden (Lernen!)
        model.apply_dopamine_learning(history, rewards, learning_rate=0.5, decay=0.01)
        
        # --- 4. VISUALISIERUNG (Alle 5 Batches updaten) ---
        if batch_idx % 5 == 0:
            # 4.1 Bild & Vorhersage
            ax_img.clear()
            img_display = data[0].cpu().view(28, 28).numpy()
            ax_img.imshow(img_display, cmap='gray')
            color = 'green' if pred[0] == target[0] else 'red'
            ax_img.set_title(f"Pred: {classes[pred[0]]}\nTrue: {classes[target[0]]}", color=color)
            ax_img.axis('off')
            
            # 4.2 Accuracy Kurve
            ax_acc.clear()
            ax_acc.plot(acc_history, color='cyan', linewidth=2)
            ax_acc.set_title("Moving Average Accuracy (Synaptic Learning Only)")
            ax_acc.set_ylim(0, 100)
            ax_acc.grid(True, alpha=0.3)
            ax_acc.set_facecolor('#1e1e1e')
            
            # 4.3 Synapsen Heatmap (Wer spricht mit wem?)
            ax_heat.clear()
            mask_matrix = np.zeros((model.num_areas, model.num_areas))
            for out_a in range(model.num_areas):
                for in_a in range(model.num_areas):
                    mask_matrix[out_a, in_a] = model.connections[f"{in_a}_to_{out_a}"].mask.mean().item()
            
            cax = ax_heat.imshow(mask_matrix, cmap='magma', vmin=0, vmax=1)
            ax_heat.set_title("Synaptic Mask Strength (0 to 1)")
            ax_heat.set_xlabel("From Area")
            ax_heat.set_ylabel("To Area")
            ax_heat.set_xticks(range(4))
            ax_heat.set_yticks(range(4))
            
            # 4.4 Aktivität pro Tick
            ax_act.clear()
            activities = np.zeros((model.num_ticks, model.num_areas))
            for t in range(model.num_ticks):
                for a in range(model.num_areas):
                    activities[t, a] = history[t+1][a].abs().mean().item()
            
            for a in range(model.num_areas):
                ax_act.plot(range(1, model.num_ticks+1), activities[:, a], marker='o', label=f'Area {a}')
            ax_act.set_title("Neural Activity Level per Tick")
            ax_act.set_xlabel("Time Tick")
            ax_act.set_ylabel("Mean Activation")
            ax_act.set_xticks(range(1, model.num_ticks+1))
            ax_act.legend()
            ax_act.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.pause(0.01) # UI Event Loop aktualisieren

if __name__ == "__main__":
    run_laboratory()