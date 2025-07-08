
import tkinter as tk
import time
import threading

class CounterWindow:
    def __init__(self, master):
        self.master = master
        master.title("Fenêtre avec compteur")
        
        # Configuration de la taille de la fenêtre
        master.geometry("400x200")
        
        # Texte à afficher
        self.label_text = tk.Label(master, text="Compteur d'engueulades", font=('Arial', 14))
        self.label_text.pack(pady=20)
        
        # Compteur
        self.counter = 0
        self.label_counter = tk.Label(master, text=str(self.counter), font=('Arial', 24, 'bold'))
        self.label_counter.pack()
        
        # Bouton pour démarrer/arrêter
        self.running = False
        self.start_stop_button = tk.Button(master, text="Démarrer", command=self.toggle_counter)
        self.start_stop_button.pack(pady=20)
    
    def toggle_counter(self):
        if self.running:
            self.running = False
            self.start_stop_button.config(text="Démarrer")
        else:
            self.running = True
            self.start_stop_button.config(text="Arrêter")
            # Démarrer le compteur dans un thread séparé
            threading.Thread(target=self.run_counter, daemon=True).start()
    
    def run_counter(self):
        while self.running:
            time.sleep(1.1)  # Intervalle de temps entre les incréments
            self.counter += 1
            # Mettre à jour l'interface graphique depuis le thread principal
            self.master.after(0, self.update_counter)
    
    def update_counter(self):
        self.label_counter.config(text=str(self.counter))

# Créer et lancer la fenêtre
root = tk.Tk()
app = CounterWindow(root)
app.toggle_counter()
root.mainloop()
