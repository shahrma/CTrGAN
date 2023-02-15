"""Create a pyplot plot and save to buffer."""

import io
import matplotlib.pyplot as plt
import PIL.Image
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
from math import ceil,sqrt
import torchvision.transforms.functional as F
import numpy as np


def imshowpts(im,pts=[],figure_flag = True, opts =[],title = '',show_flag = True) :
    if figure_flag :
        plt.figure()
    if im.shape[0] == 3 or im.shape[0] == 1 or im.shape[0] == 4:
        plt.imshow(im.permute(1,2,0))
    else :
        plt.imshow(im)
    if len(pts) != 0:
        plt.scatter(pts[:, 0], pts[:, 1], s=10, marker='.', c='r') # pts[:, 0] => x , pts[:, 1] => y
        plt.scatter(pts[0, 0], pts[0, 1], s=30, marker='*', c='r')
    if len(opts) != 0:
        plt.scatter(opts[:, 0], opts[:, 1], s=10, marker='o', c='g') # pts[:, 0] => x , pts[:, 1] => y
        plt.scatter(pts[0, 0], pts[0, 1], s=30, marker='*', c='g')
    plt.pause(0.001)  # p
    plt.title(title)
    if show_flag :
        plt.show(block=False)

def fig_to_image():
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)

    return image

class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')

# now use it as the replacement of transforms.Pad class

def get_grid_image(gen, point):
    image_size = 256
    data_transforms = transforms.Compose([
        SquarePad(),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    im_= PIL.Image.fromarray(np.uint8( gen[0]))

    images = [data_transforms(image) for image in gen]
    images = torchvision.progressive_upscaling(images)
    images = list(map(lambda x: x.squeeze(dim=0), images))
    image = torchvision.make_grid(        images,nrow=int(ceil(sqrt(len(images))))
    )
    return image.cpu().numpy().transpose(1, 2, 0)

'''
plt.figure()
plt.plot([1, 2])
plt.title("test")



# Prepare the plot
plot_buf = gen_plot()



writer = SummaryWriter(comment='hello imaage')
#x = torchvision.utils.make_grid(image, normalize=True, scale_each=True)
for n_iter in range(100):
    if n_iter % 10 == 0:
        writer.add_image('Image', image, n_iter)

'''