import torch
from infer_caption import pred_vec_to_text
from PIL import ImageDraw
from PIL import ImageFont
import torchvision
from matplotlib.pyplot import imshow
import numpy as np
import cv2
from PIL import Image

to_pil = torchvision.transforms.ToPILImage(mode='RGB')

def create_meme(generator,
        encoder,
        decoder,
        data_loader,
        device,
        dataset,
        random_noise=None
    ):

    if random_noise is None:
        random_noise = torch.nn.init.trunc_normal_(
            torch.zeros((1, 100)),
            a=-2,
            b=2,
        )
    

    meme_background = generator.forward(*generator.sampler(1, 'cpu'))

    features = encoder(meme_background)

    _, dummy_cap = next(data_loader)
    dummy_cap = dummy_cap[:1].to(device)
    caption = decoder(features, dummy_cap)

    cap = " ".join([word for word in pred_vec_to_text(caption, dataset)[0] if word != "<UNK>"])
    image = meme_background[0].permute(1, 2, 0).detach().numpy()
    image = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    image = image.resize((256, 256))

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("SF-Compact-Display-Bold.otf", 16)
    # draw.text((0, 0),"Sample Text",(255,255,255),font=font)

    def drawTextWithOutline(text, x, y):
        draw.text((x-2, y-2), text,(0,0,0),font=font)
        draw.text((x+2, y-2), text,(0,0,0),font=font)
        draw.text((x+2, y+2), text,(0,0,0),font=font)
        draw.text((x-2, y+2), text,(0,0,0),font=font)
        draw.text((x, y), text, (255,255,255), font=font)

    def drawText(img, text, pos):
        text = text.upper()
        w, h = draw.textsize(text, font) # measure the size the text will take

        lineCount = 1
        if w > img.width:
            lineCount = int(round((w / img.width) + 1))

        lines = []
        if lineCount > 1:

            lastCut = 0
            isLast = False
            for i in range(0,lineCount):
                if lastCut == 0:
                    cut = (len(text) // lineCount) * i
                else:
                    cut = lastCut

                if i < lineCount-1:
                    nextCut = (len(text) // lineCount) * (i+1)
                else:
                    nextCut = len(text)
                    isLast = True

                # make sure we don't cut words in half
                if nextCut == len(text) or text[nextCut] == " ":
                    pass
                else:
                    while text[nextCut] != " ":
                        nextCut += 1

                line = text[cut:nextCut].strip()

                # is line still fitting ?
                w, h = draw.textsize(line, font)
                if not isLast and w > img.width:
                    nextCut -= 1
                    while text[nextCut] != " ":
                        nextCut -= 1

                lastCut = nextCut
                lines.append(text[cut:nextCut].strip())

        else:
            lines.append(text)

        lastY = -h
        if pos == "bottom":
            lastY = img.height - h * (lineCount+1) - 10

        for i in range(0, lineCount):
            w, h = draw.textsize(lines[i], font)
            x = img.width/2 - w/2
            y = lastY + h
            drawTextWithOutline(lines[i], x, y)
            lastY = y

    drawText(image, cap, "top")

    return image

    