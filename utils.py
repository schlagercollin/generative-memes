import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import textwrap

def get_preprocessing_normalisation_transform(image_size):
    return T.Compose([
        T.ConvertImageDtype(torch.uint8),
        T.AutoAugment(),
        T.Resize(image_size),
        T.ConvertImageDtype(torch.float),
        T.CenterCrop(image_size),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def tensor_to_image(output: torch.Tensor, ncol: int=4, padding: int=2) -> Image:
    """Convert the tensor-based output from a Generator into a PIL image.
    
    Note: use np.asarray(img) to convert to a numpy array.

    Args:
        output (torch.Tensor): output from Generator model
        ncol (int, optional): number of columns. Defaults to 4.
        padding (int, optional): padding around each image. Defaults to 2.

    Returns:
        Image: PIL image object
    """
    out = torchvision.utils.make_grid(output, nrow=ncol, normalize=True, padding=padding)
    return T.ToPILImage()(out)


def interpolate(a: torch.Tensor, b: torch.Tensor, num_steps: int=64) -> torch.Tensor:
    """Linearly interpolate between tensors A and B with num_steps steps.

    Args:
        a (torch.Tensor): Starting point.
        b (torch.Tensor): Ending point.
        num_points (int, optional): Number of intermediate tensors to create. Defaults to 64.

    Returns:
        torch.Tensor: (num_steps, **a.shape) sized tensor.
    """
    
    # TODO: add different types of interpolation
    
    if a.shape != b.shape:
        raise ValueError(f"a and b need the same shape. got {a.shape} and {b.shape}.")
    
    return torch.stack([torch.lerp(a, b, weight) for weight in torch.linspace(0, 1, num_steps)])


class Meme:
    """
    Adapted from: 
    https://stackoverflow.com/questions/63498671/create-a-captioned-meme-using-python-and-pil
    """
    basewidth = 1200            #Width to make the meme
    fontBase = 100              #Font size
    letSpacing = 9              #Space between letters
    fill = (255, 255, 255)      #TextColor
    stroke_fill = (0,0,0)       #Color of the text outline
    lineSpacing = 10            #Space between lines
    stroke_width=9              #How thick the outline of the text is
    fontfile = './Impact.ttf'   #Points to the font file (local to GitHub)

    def __init__(self, caption, image):
        self.img = self.createImage(image)
        self.d = ImageDraw.Draw(self.img)

        self.splitCaption = textwrap.wrap(caption, width=20)  # The text can be wider than the img. If thats the case split the text into multiple lines
        self.splitCaption.reverse()                           # Draw the lines of text from the bottom up

        fontSize = self.fontBase+10 if len(self.splitCaption) <= 1 else self.fontBase   #If there is only one line, make the text a bit larger
        self.font = ImageFont.truetype(font=self.fontfile, size=fontSize)
        # self.shadowFont = ImageFont.truetype(font='./impact.ttf', size=fontSize+10)

    def draw(self):
        '''
        Draws text onto this objects img object
        :return: A pillow image object with text drawn onto the image
        '''
        (iw, ih) = self.img.size
        (_, th) = self.d.textsize(self.splitCaption[0], font=self.font) #Height of the text
        # y = (ih - (ih / 10)) - (th / 2) #The starting y position to draw the last line of text. Text in drawn from the bottom line up
        # y = (0 + (ih / 10)) + (th / 2)  # Collin edit: flip signs to start text at top
        y = 5  # Collin edit: this starting point seems reasonable

        for cap in self.splitCaption:   #For each line of text
            (tw, _) = self.d.textsize(cap, font=self.font)  # Getting the position of the text
            x = ((iw - tw) - (len(cap) * self.letSpacing))/2  # Center the text and account for the spacing between letters

            self.drawLine(x=x, y=y, caption=cap)
            # y = y - th - self.lineSpacing  # Next block of text is higher up
            y = y + th + self.lineSpacing # Collin edit: flip signs to start text at top

        wpercent = ((self.basewidth/2) / float(self.img.size[0]))
        hsize = int((float(self.img.size[1]) * float(wpercent)))
        return self.img.resize((int(self.basewidth/2), hsize))

    def createImage(self, img):
        '''
        Resizes the image to a resonable standard size
        :param image: Path to an image file or PIL Image
        :return: A pil image object
        '''
        if type(img) == str:
            img = Image.open(img)
        wpercent = (self.basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        return img.resize((self.basewidth, hsize))

    def drawLine(self, x, y, caption):
        '''
        The text gets split into multiple lines if it is wider than the image. This function draws a single line
        :param x: The starting x coordinate of the text
        :param y: The starting y coordinate of the text
        :param caption: The text to write on the image
        :return: None
        '''
        for idx in range(0, len(caption)):  #For each letter in the line of text
            char = caption[idx]
            w, h = self.font.getsize(char)  #width and height of the letter
            self.d.text(
                (x, y),
                char,
                fill=self.fill,
                stroke_width=self.stroke_width,
                font=self.font,
                stroke_fill=self.stroke_fill
            )  # Drawing the text character by character. This way spacing can be added between letters
            x += w + self.letSpacing #The next character must be drawn at an x position more to the right

if __name__ == '__main__':
    arr = np.load('losses.npy')
    x = np.arange(0, arr.shape[0])
    plt.plot(x, arr)
    plt.show()