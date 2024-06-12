from lightning import LightningApp, LightningWork

class GPUApp(LightningWork):
    def run(self):
        import subprocess
        subprocess.run(["streamlit", "run", "Home.py", "-- --device", "cuda"])

app = LightningApp(GPUApp())
