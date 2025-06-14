import sys
import os
import tkinter as tk
from tkinter import messagebox
from ui.gui_main import MetroProjectMainGUI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    try:
        root = tk.Tk()
        root.title("Projet Métro Parisien - team12")
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
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"Erreur critique: {e}")
        sys.exit(1) 