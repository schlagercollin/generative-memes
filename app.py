import torch
import torch

from flask import Flask, render_template, request
from inference import load_inference_components, create_meme_image
from utils import Meme

from PIL import Image
import base64
import io

app = Flask(__name__)
generator, model, data_loader, device, dataset = load_inference_components()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        
        input_val = float(request.form.get("input_val"))
        
        # just set the seed and generate that way
        torch.manual_seed(input_val)
        im = create_meme_image(generator, "cpu")
        im = Meme("Funny caption", im).draw()
        
        data = io.BytesIO()
        im.save(data, "JPEG")
        encoded_img_data = base64.b64encode(data.getvalue())
        
        return render_template("alt.html", img_data=encoded_img_data.decode('utf-8'))
   
    return render_template("alt.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)