from PIL import Image
import numpy as np
# extract patches from 3 size of img, 1200x900, 600x600, 300x300
# total 12 + 6 + 1 patches
class PatchExtractor:
    def __init__(self, img, patch_size, stride,noIP=False):
        if type(img) == np.ndarray:
            img = Image.fromarray(img)

        self.img0 = img
        if noIP:
            self.img_list = [self.img0]
        else:
            self.img1 = img.resize((600, 600), Image.BILINEAR)
            self.img2 = img.resize((300, 300), Image.BILINEAR)
            self.img_list = [self.img0, self.img1, self.img2]
        # self.img_list = [self.img0]
        self.size = patch_size
        self.stride = stride

    def extract_patches(self):

        patches = []
        for im in self.img_list:
            wp, hp = self.shape(im)
            temp = [self.extract_patch(im, (w, h)) for h in range(hp) for w in range(wp)]
            patches.extend(temp)
        return patches

    def extract_patch(self, img, patch):

        return img.crop((
            patch[0] * self.stride,  # left
            patch[1] * self.stride,  # up
            patch[0] * self.stride + self.size,  # right
            patch[1] * self.stride + self.size  # down
        ))

    def shape(self, img):
        wp = int((img.width - self.size) / self.stride + 1)
        hp = int((img.height - self.size) / self.stride + 1)
        return wp, hp


