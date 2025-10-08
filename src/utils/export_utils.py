import io
import base64
import pickle
import torch
import torch.nn as nn

def get_model_download_link(model, model_name):
    """Generate download link for trained models"""
    if isinstance(model, nn.Module):
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        return f'<a href="data:application/octet-stream;base64,{b64}" download="{model_name}.pth">Download Model</a>'
    else:
        b = io.BytesIO()
        pickle.dump(model, b)
        b_val = b.getvalue()
        b64 = base64.b64encode(b_val).decode()
        return f'<a href="data:application/octet-stream;base64,{b64}" download="{model_name}.pkl">Download Model</a>'