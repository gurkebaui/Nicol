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
# 1. DAS BIOLOGISCHE KONTINUUM-GEHIRN
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

class BiologicalBrain(nn.Module):
    def __init__(self, checkpoint_path, num_areas=4):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_areas = num_areas
        self.threshold = 0.02
        
        print("Lade Gehirn und formatiere Nested-Learning-Zonen...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model_state = ckpt['model_state']
        
        # Clustering wie gehabt
        kmeans = KMeans(n_clusters=num_areas, random_state=42, n_init=10)
        self.neuron_labels = kmeans.fit_predict(ckpt['co_activation'].numpy())
        
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
        self.long_term_corr = {} # Sammler für den "Schlaf"
        
        for out_area in range(num_areas):
            for in_area in range(num_areas):
                idx_out = getattr(self, f"area_idx_{out_area}")
                idx_in = getattr(self, f"area_idx_{in_area}")
                w_block = full_synapses_w[idx_out][:, idx_in]
                b_block = full_synapses_b[idx_out] if in_area == out_area else torch.zeros(len(idx_out))
                
                mask_init = torch.ones_like(w_block) * (0.8 if out_area == in_area else 0.1)
                conn_name = f"{in_area}_to_{out_area}"
                
                self.connections[conn_name] = AreaConnection(w_block, b_block, mask_init)
                self.long_term_corr[conn_name] = torch.zeros_like(w_block, device=self.device)

        # --- Die Nested Learning Hierarchie (CLS Theorie) ---
        # Definiert, welches Areal sich wie verhält (angewandt auf dendritische Eingänge -> out_area)
        self.nested_config = {
            0: {'name': 'Hippocampus', 'mask_lr': 0.200, 'weight_lr': 0.050, 'sleep_cycle': 50},
            1: {'name': 'Neocortex_1', 'mask_lr': 0.050, 'weight_lr': 0.010, 'sleep_cycle': 200},
            2: {'name': 'Neocortex_2', 'mask_lr': 0.050, 'weight_lr': 0.010, 'sleep_cycle': 200},
            3: {'name': 'Brainstem',   'mask_lr': 0.005, 'weight_lr': 0.002, 'sleep_cycle': 1000},
        }

        self.Z = None
        self.Z_prev = None
        self.F = None # FATIGUE (Ermüdung für Gehirnwellen!)
        self.retina_strength = 1.0
        self.to(self.device)

    def step(self, x, is_new_image=False):
        batch_size = x.size(0)
        
        # Initialisierung beim ersten Aufruf
        if self.Z is None:
            self.Z =[torch.zeros(batch_size, getattr(self, f"area_idx_{i}").size(0), device=self.device) for i in range(self.num_areas)]
            self.Z_prev =[torch.zeros_like(z) for z in self.Z]
            self.F =[torch.zeros_like(z) for z in self.Z] # Erschöpfung startet bei 0

        # --- 1. Netzhaut-Adaption ---
        if is_new_image:
            self.retina_strength = 1.0 # Voller Schock beim neuen Bild
        else:
            # Netzhaut gewöhnt sich an das Bild, Reiz wird schwächer, 
            # das Gehirn "übernimmt" die interne Diskussion.
            self.retina_strength = max(0.1, self.retina_strength - 0.1)

        # Z_prev speichern für Hebbian Learning
        for i in range(self.num_areas):
            self.Z_prev[i] = self.Z[i].clone()

        stimulus =[]
        for i in range(self.num_areas):
            idx = getattr(self, f"area_idx_{i}")
            stim = F.linear(x, self.receptors_w[idx], self.receptors_b[idx]) * self.retina_strength
            stimulus.append(stim)

        # --- 2. Das Routing (Das Gehirn spricht miteinander) ---
        Z_next =[torch.zeros_like(z) for z in self.Z]
        for out_area in range(self.num_areas):
            for in_area in range(self.num_areas):
                source_signal = self.Z[in_area]
                if source_signal.abs().mean().item() > self.threshold:
                    Z_next[out_area] += self.connections[f"{in_area}_to_{out_area}"](source_signal)

        # --- 3. Neuronale Ermüdung & Oszillation (Die Magie!) ---
        for i in range(self.num_areas):
            # Fatigue baut sich auf, wenn Z stark ist (vorzeichenbehaftet!)
            self.F[i] = self.F[i] * 0.8 + self.Z[i] * 0.4
            
            # Tanh begrenzt auf -1 bis 1. 
            # Die Fatigue zieht das Neuron in die gegenteilige Richtung!
            self.Z[i] = torch.tanh(Z_next[i] + stimulus[i] - self.F[i])
            
        # --- 4. Auslesen der Gedanken & des Speak-Gates ---
        Z_flat = torch.zeros(batch_size, sum(len(getattr(self, f"area_idx_{i}")) for i in range(self.num_areas)), device=self.device)
        for i in range(self.num_areas):
            Z_flat[:, getattr(self, f"area_idx_{i}")] = self.Z[i]
            
        raw_out = F.linear(Z_flat, self.output_lobe_w, self.output_lobe_b)
        logits = raw_out[:, :10]
        gate = torch.sigmoid(raw_out[:, 10])
        
        return logits, gate

    def apply_nested_learning(self, rewards, global_tick):
        """
        Das 4-stufige Gedächtnis-Kontinuum (Ohne Backprop, rein organisch).
        rewards:[Batch_Size, 1] Tensor.
        """
        for out_area in range(self.num_areas):
            conf = self.nested_config[out_area]
            
            for in_area in range(self.num_areas):
                conn_name = f"{in_area}_to_{out_area}"
                conn = self.connections[conn_name]
                
                pre_act = self.Z_prev[in_area]
                post_act = self.Z[out_area]
                
                # Hebb'sche Korrelation modulatiert durch Dopamin (Reward)
                weighted_post = post_act * rewards
                correlation = torch.matmul(weighted_post.T, pre_act) / pre_act.size(0)
                
                # ----------------------------------------------------
                # LEVEL 1: Kurzzeitgedächtnis (Plastizität der Masken)
                # Passiert JEDEN TICK.
                # ----------------------------------------------------
                noise = torch.randn_like(conn.mask) * 0.005
                new_mask = conn.mask + (conf['mask_lr'] * correlation) - (0.005 * conn.mask) + noise
                conn.mask.data = torch.clamp(new_mask, 0.0, 1.0)
                
                # Sammle das Wissen für den Schlaf!
                self.long_term_corr[conn_name] += correlation.detach()
                
                # ----------------------------------------------------
                # LEVEL 2-4: Micro-Schlaf / Konsolidierung in Basis-Wissen
                # Passiert nur, wenn der Sleep Cycle des Areals erreicht ist!
                # ----------------------------------------------------
                if global_tick > 0 and global_tick % conf['sleep_cycle'] == 0:
                    with torch.no_grad():
                        # SANFTES Umschreiben der harten Gewichte! (Memory Consolidation)
                        conn.weight.data += conf['weight_lr'] * self.long_term_corr[conn_name]
                        
                        # Die tiefen Gewichte haben es jetzt verstanden ->
                        # Kurzzeitgedächtnis (Masken) wird entspannt, Platz für Neues.
                        conn.mask.data = conn.mask.data * 0.90
                        
                        # Zähler zurücksetzen
                        self.long_term_corr[conn_name].zero_()

# ==========================================
# 2. DER VISUALIZER (EEG-Wellen Labor)
# ==========================================
def run_eeg_laboratory():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starte biologisches EEG Labor auf: {device}")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    loader_iter = iter(loader)

    # Lade dein kontinuierlich trainiertes Modell (aus Phase 1)
    model = BiologicalBrain("god_phase1_continuous.pt", num_areas=4)
    model.eval()
    
    plt.ion()
    fig = plt.figure(figsize=(15, 8))
    fig.canvas.manager.set_window_title('Nova EEG: Brain Waves & Sleep Cycles')
    
    ax_img = plt.subplot2grid((2, 3), (0, 0))
    ax_acc = plt.subplot2grid((2, 3), (0, 1), colspan=2)
    ax_act = plt.subplot2grid((2, 3), (1, 0), colspan=3)
    
    HISTORY_LEN = 150
    area_histories = [deque([0.0]*HISTORY_LEN, maxlen=HISTORY_LEN) for _ in range(4)]
    speak_markers_x, speak_markers_y, acc_history = [], [],[]
    
    global_tick = 0
    ticks_per_image = 25 # Etwas mehr Zeit pro Bild, um die Wellen zu sehen!
    current_img, current_target = next(loader_iter)
    current_img, current_target = current_img.to(device), current_target.to(device)
    moving_acc = 0.1

    print("\nSystem läuft! Beobachte, wie das Gehirn jetzt Welle schlägt.")
    
    while plt.fignum_exists(fig.number):
        global_tick += 1
        is_new = (global_tick % ticks_per_image == 1)
        
        if is_new and global_tick > 1:
            try:
                current_img, current_target = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                current_img, current_target = next(loader_iter)
            current_img, current_target = current_img.to(device), current_target.to(device)

        # 1. EIN GEDANKEN-TICK
        logits, gate = model.step(current_img, is_new_image=is_new)
        
        pred = logits.argmax(dim=1)
        # Dopamin berechnen: Richtig = +1.0, Falsch = -0.5
        rewards = torch.where(pred == current_target, torch.tensor(1.0, device=device), torch.tensor(-0.5, device=device)).unsqueeze(1)
        
        # 2. DAS NESTED LEARNING (Plastizität & Schlaf)
        model.apply_nested_learning(rewards, global_tick)
        
        if is_new:
            # Accuracy tracken (Wir messen sie in dem Moment, wo ein neues Bild kommt)
            batch_acc = (pred == current_target).float().mean().item()
            moving_acc = 0.9 * moving_acc + 0.1 * batch_acc
            acc_history.append(moving_acc * 100)

        # Tracking für Plot
        for i in range(4):
            area_histories[i].append(model.Z[i].mean().item()) # .abs() entfernt, damit wir ECHTE Wellen (- bis +) sehen!

        if gate.mean().item() > 0.85: 
            speak_markers_x.append(HISTORY_LEN - 1)
            speak_markers_y.append(model.Z[1].mean().item()) # Markiere auf einer Welle
            
        speak_markers_x =[x - 1 for x in speak_markers_x if x > 0]
        speak_markers_y = speak_markers_y[-len(speak_markers_x):]

        # 3. GUI UPDATE (Smooth 30 FPS)
        if global_tick % 3 == 0:
            ax_img.clear()
            ax_img.imshow(current_img[0].cpu().view(28, 28).numpy(), cmap='gray')
            ax_img.axis('off')
            if gate[0].item() > 0.85:
                color = '#00ff00' if pred[0] == current_target[0] else '#ff0000'
                ax_img.set_title(f"OUTPUT: {pred[0].item()} !!!", color=color, fontsize=18, fontweight='bold')
            else:
                ax_img.set_title("Thinking...", color='#888888', fontsize=14)

            ax_acc.clear()
            ax_acc.plot(acc_history, color='magenta', linewidth=2)
            ax_acc.set_ylim(0, 100)
            ax_acc.set_title(f"Continuous Area Accuracy | Tick: {global_tick}")
            ax_acc.grid(True, alpha=0.3)
            ax_acc.set_facecolor('#1e1e1e')
            
            ax_act.clear()
            colors =['#00e5ff', '#ffaa00', '#00ff00', '#ff0055']
            names =['Hippocampus', 'Neocortex 1', 'Neocortex 2', 'Brainstem']
            for i in range(4):
                ax_act.plot(area_histories[i], label=f'{names[i]} (Areal {i})', color=colors[i], linewidth=1.5)
            
            time_since_switch = global_tick % ticks_per_image
            pos_of_switch = HISTORY_LEN - 1 - time_since_switch
            if 0 < pos_of_switch < HISTORY_LEN:
                ax_act.axvline(x=pos_of_switch, color='white', linestyle=':', alpha=0.5)
                
            if speak_markers_x:
                ax_act.scatter(speak_markers_x, speak_markers_y, color='white', s=80, marker='*', zorder=5)

            ax_act.set_ylim(-1.0, 1.0) # Wellen gehen jetzt ins Minus!
            ax_act.set_title("Live EEG: Biological Oscillations (Fatigue-driven)")
            ax_act.legend(loc="upper left")
            ax_act.set_facecolor('#1e1e1e')
            ax_act.axis('off')

            plt.tight_layout()
            plt.pause(0.001)

if __name__ == "__main__":
    run_eeg_laboratory()