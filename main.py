import sys
import os
import tkinter as tk
from tkinter import messagebox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from gui_visualizer import MetroProjectMainGUI
except ImportError as e:
    print(f" Erreur d'import: {e}")
    sys.exit(1)

def main():
    """Fonction principale du projet"""
    try:
        root = tk.Tk()
        root.title("Projet Métro Parisien - TEAM1")
        root.eval('tk::PlaceWindow . center')
        app = MetroProjectMainGUI(root)
        root.mainloop()
    
        return 0
        
    except Exception as e:
        error_msg = f" Erreur lors du lancement: {str(e)}"
        print(error_msg)
        
        try:
            messagebox.showerror("Erreur", error_msg)
        except:
            pass
        
        return 1


if __name__ == "__main__":
    """Point d'entrée du script"""
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"Erreur critique: {e}")
        sys.exit(1) 