import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Wir laden nur kurz die Gewichte, wir brauchen nicht das ganze komplexe Routing-Modell für den Scanner
class BrainScanner:
    def __init__(self, checkpoint_path, num_areas=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Lade Gehirn für den MRT-Scan...")
        
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model_state = ckpt['model_state']
        co_activation = ckpt['co_activation'].numpy()
        
        self.receptors_w = model_state['receptors.weight'] # Shape: [2048, 784]
        
        print("Berechne Areale...")
        kmeans = KMeans(n_clusters=num_areas, random_state=42, n_init=10)
        self.neuron_labels = kmeans.fit_predict(co_activation)
        
        self.areas =[]
        for i in range(num_areas):
            idx = np.where(self.neuron_labels == i)[0]
            self.areas.append(idx)
            
    def get_area_receptive_field(self, area_idx):
        """Berechnet das durchschnittliche 'Auge' eines ganzen Areals"""
        neurons_in_area = self.areas[area_idx]
        # Durchschnitt über alle Rezeptor-Gewichte dieses Areals
        avg_weights = self.receptors_w[neurons_in_area].mean(dim=0)
        return avg_weights.view(28, 28).numpy()

    def get_loudest_neurons(self, image_tensor, top_k=5):
        """Findet die Neuronen, die beim Anblick des Bildes am stärksten feuern"""
        # image_tensor shape: [784]
        # Wir berechnen die Aktivierung nach Tick 0 (Direkter Input)
        activations = torch.tanh(F.linear(image_tensor.unsqueeze(0), self.receptors_w))[0]
        
        # Finde die Top K stärksten Neuronen
        top_activations, top_indices = torch.topk(activations, top_k)
        
        loud_neurons = []
        for i in range(top_k):
            idx = top_indices[i].item()
            act = top_activations[i].item()
            # Finde heraus, zu welchem Areal das Neuron gehört
            area = self.neuron_labels[idx]
            # Hole das visuelle Muster, auf das dieses Neuron reagiert
            rf = self.receptors_w[idx].view(28, 28).numpy()
            loud_neurons.append({'idx': idx, 'activation': act, 'area': area, 'rf': rf})
            
        return loud_neurons

def run_scanner():
    scanner = BrainScanner("god_phase1.pt", num_areas=4)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    # Wir nehmen den normalen MNIST Test-Satz, um zu sehen, was er bei Ziffern macht
    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    plt.ion()
    fig = plt.figure(figsize=(16, 10))
    fig.canvas.manager.set_window_title('Nova MRI: Deep Brain Scanner')

    for batch_idx, (data, target) in enumerate(loader):
        img_flat = data[0]
        img_2d = img_flat.view(28, 28).numpy()
        
        fig.clf()
        
        # 1. Das Originalbild
        ax_img = fig.add_subplot(2, 6, 1)
        ax_img.imshow(img_2d, cmap='gray')
        ax_img.set_title(f"Input: {target.item()}")
        ax_img.axis('off')
        
        # 2. Die Receptive Fields der Areale (Was sucht die Region im Durchschnitt?)
        for i in range(4):
            ax_area = fig.add_subplot(2, 6, 3 + i)
            rf_area = scanner.get_area_receptive_field(i)
            # RdBu Colormap: Rot = "Ich will Pixel", Blau = "Pixel verbieten"
            vmax = np.max(np.abs(rf_area))
            ax_area.imshow(rf_area, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax_area.set_title(f"Average Eye\nArea {i}")
            ax_area.axis('off')
            
        # 3. Die 5 "lautesten" Einzel-Neuronen für DIESES Bild
        loud_neurons = scanner.get_loudest_neurons(img_flat, top_k=5)
        
        for i, neuron in enumerate(loud_neurons):
            ax_neuron = fig.add_subplot(2, 5, 6 + i)
            rf = neuron['rf']
            vmax = np.max(np.abs(rf))
            ax_neuron.imshow(rf, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax_neuron.set_title(f"Neuron {neuron['idx']} (Area {neuron['area']})\nAct: {neuron['activation']:.2f}")
            ax_neuron.axis('off')
            
        plt.suptitle("Deep Brain Scan: Was sehen die Rezeptoren?", fontsize=16)
        plt.tight_layout()
        plt.pause(0.1)
        
        input("Drücke ENTER in der Konsole für das nächste Bild (oder Strg+C zum Beenden)...")

if __name__ == "__main__":
    run_scanner()