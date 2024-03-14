import tkinter as tk
from tkinter import scrolledtext

class LLM_GUI:
    def __init__(self, master):
        self.master = master
        master.title("LLM Interaction")

        # main view
        self.display_text = scrolledtext.ScrolledText(master, height=10, bg="black")
        self.display_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.display_text.tag_configure('user', foreground='white')
        self.display_text.tag_configure('ai', foreground='lightgreen')
        self.display_text.config(state=tk.DISABLED) 

        # upload_button
        self.upload_button = tk.Button(master, text="≡", command=self.upload_data)
        self.upload_button.pack(side=tk.LEFT, padx=(10, 0), pady=(0, 10))

        # user_input
        self.user_input = tk.Entry(master)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10), pady=(0, 10))

        # send_button
        self.send_button = tk.Button(master, text="↑", command=self.send_query)
        self.send_button.pack(side=tk.LEFT, padx=(0, 10), pady=(0, 10))

    def upload_data(self):
        pass

    def send_query(self):
        query = self.user_input.get()
        if query:
            self.insert_text(f"USER: {query}\n", 'user')
            self.user_input.delete(0, tk.END)  # clean input

            self.insert_text("Model is processing...\n", 'ai')
            self.master.after(1000, self.model_reply, query)  

    def model_reply(self, query):
        processing_index = self.display_text.search("Model is processing...", "1.0", tk.END)
        if processing_index:
            self.display_text.config(state=tk.NORMAL)
            self.display_text.delete(f"{processing_index} linestart", f"{processing_index} lineend + 1 chars")
            self.display_text.config(state=tk.DISABLED)

        # model reply
        self.insert_text(f"AI: This is a reply to '{query}'.\n", 'ai')

    def insert_text(self, text, tag):
        self.display_text.config(state=tk.NORMAL)  # insert text
        self.display_text.insert(tk.END, text, tag)
        self.display_text.see(tk.END)  # roll to buttom
        self.display_text.config(state=tk.DISABLED)  # insert text disabled

def main():
    root = tk.Tk()
    app = LLM_GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
