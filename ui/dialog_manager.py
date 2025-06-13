import tkinter as tk
from tkinter import ttk


class DialogManager:
    def __init__(self, parent_window):
        self.parent = parent_window
    
    def show_resize_configuration_dialog(self):
        dialog = tk.Toplevel(self.parent)
        dialog.title("Configuration Challenge + Redimensionnement")
        dialog.geometry("500x500")
        dialog.resizable(False, False)
        dialog.transient(self.parent)
        dialog.grab_set()
        
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (250)
        y = (dialog.winfo_screenheight() // 2) - (175)
        dialog.geometry(f"500x500+{x}+{y}")
        
        result = {'cancelled': True}
        
        header_frame = ttk.Frame(dialog)
        header_frame.pack(fill=tk.X, padx=20, pady=20)
        
        ttk.Label(header_frame, text="CONFIGURATION CHALLENGE", 
                 font=("Arial", 14, "bold")).pack(anchor=tk.W)
        
        ttk.Label(header_frame, text="Dossier: challenge/BD_CHALLENGE", 
                 font=("Arial", 10), foreground="gray").pack(anchor=tk.W, pady=(5, 0))
        
        ttk.Label(header_frame, text="Vérités terrain: Détection automatique fichier .mat", 
                 font=("Arial", 10), foreground="gray").pack(anchor=tk.W, pady=(2, 0))
        
        ttk.Separator(dialog, orient='horizontal').pack(fill=tk.X, padx=20, pady=15)
        
        resize_frame = ttk.LabelFrame(dialog, text="Facteur de Redimensionnement", padding="20")
        resize_frame.pack(fill=tk.X, padx=20, pady=10)
        
        resize_var = tk.DoubleVar(value=1.0)
        
        ttk.Label(resize_frame, text="Facteur de redimensionnement:", 
                 font=("Arial", 11)).pack(anchor=tk.W, pady=(0, 10))
        
        scale_frame = ttk.Frame(resize_frame)
        scale_frame.pack(fill=tk.X, pady=10)
        
        scale = ttk.Scale(scale_frame, from_=0.1, to=2.0, variable=resize_var, 
                         orient=tk.HORIZONTAL, length=400)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        scale_label = ttk.Label(scale_frame, text="1.00x", width=8, 
                               font=("Arial", 11, "bold"))
        scale_label.pack(side=tk.RIGHT, padx=(15, 0))
        
        def update_scale_label(*args):
            scale_label.config(text=f"{resize_var.get():.2f}x")
        
        resize_var.trace('w', update_scale_label)
        
        preset_frame = ttk.Frame(resize_frame)
        preset_frame.pack(fill=tk.X, pady=(15, 0))
        
        ttk.Label(preset_frame, text="Presets rapides:", 
                 font=("Arial", 10)).pack(anchor=tk.W, pady=(0, 8))
        
        buttons_frame = ttk.Frame(preset_frame)
        buttons_frame.pack(fill=tk.X)
        
        def set_resize_factor(factor):
            resize_var.set(factor)
        
        for factor in [0.5, 0.75, 1.0, 1.5, 2.0]:
            ttk.Button(buttons_frame, text=f"{factor}x", width=10,
                      command=lambda f=factor: set_resize_factor(f)).pack(side=tk.LEFT, padx=3)
        
        ttk.Separator(dialog, orient='horizontal').pack(fill=tk.X, padx=20, pady=15)
        
        action_frame = ttk.Frame(dialog)
        action_frame.pack(fill=tk.X, padx=20, pady=20)
        
        def on_start():
            result.update({
                'cancelled': False,
                'resize_factor': resize_var.get()
            })
            dialog.destroy()
        
        def on_cancel():
            result['cancelled'] = True
            dialog.destroy()
        
        start_button = ttk.Button(action_frame, text="COMMENCER L'ANALYSE", 
                                 command=on_start)
        start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        cancel_button = ttk.Button(action_frame, text="Annuler", 
                                  command=on_cancel)
        cancel_button.pack(side=tk.LEFT)
        
        start_button.focus_set()
        
        dialog.bind('<Return>', lambda e: on_start())
        dialog.bind('<Escape>', lambda e: on_cancel())
        
        dialog.wait_window()
        
        return None if result['cancelled'] else {
            'resize_factor': result['resize_factor']
        } 