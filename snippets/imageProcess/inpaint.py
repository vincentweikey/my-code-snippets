"""inpainting by rectangle patch scannning 
"""
import numpy as np
import cv2
from skimage.util.shape import view_as_windows
from math import ceil
import matplotlib.pyplot as plt
from PIL import Image


class Fast_Synthesis_based_inpaint:
    def __init__(self, img, inpaint_mask, sample_mask, cloth_mask, in_mirror_hor=False, in_mirror_vert=False, DEBUG=False):
        '''
        img:PIL
        inpaint_mask:PIL
        sample_mask:PIL
        '''
        self.img = np.array(img)
        self.ori_img = np.array(img).copy()
        self.sample_mask = sample_mask
        self.cloth_mask = np.array(cloth_mask.convert('L')).astype('bool')
        self.inpaint_mask = np.array(inpaint_mask.convert('L')).astype('uint8')
        # 合理的采样区域
        self.sample_mask = ~self.inpaint_mask.astype('bool') * np.array(self.sample_mask.convert('L')).astype('bool')

        if np.sum(self.inpaint_mask) == 0:  # 不希望做任何更改
            self.pass_ = True
        else:
            self.pass_ = False
            self.DEBUG = DEBUG
            self.mirror_hor = in_mirror_hor
            self.mirror_vert = in_mirror_vert

            self.img = self.img / 255.

            self.x, self.y, self.w, self.h = cv2.boundingRect(self.inpaint_mask)  # 找最小外接矩形
            #             self.img[self.y:self.y+self.h, self.x:self.x+self.w].fill(1)#填充矩形为纯白

            self.init_hyper_parameters()
            self.examplePatches = self.init_patch()
            self.canvas_with_bound = self.init_canvas()
            self.initKDtrees()

    def init_hyper_parameters(self):
        self.patchSize = int(min(18, min(self.w, self.h) // 4))  # 不大于18size of the patch (without the overlap)
        self.overlapSize = max(2, self.patchSize // 6)  # 不小于2 the overlap region

        self.searchKernelSize = self.patchSize + 2 * self.overlapSize
        self.windowStep = 1

        if self.DEBUG:
            print("patchSize: %s" % self.patchSize)
            print("overlapSize: %s" % self.overlapSize)
            print("searchKernelSize: %s" % self.searchKernelSize)
            print("windowStep: %s" % self.windowStep)

    def init_patch(self):
        self.sample_area = self.img.copy()
        self.sample_mask[self.y:self.y + self.h, self.x:self.x + self.w].fill(False)
        self.sample_area[~self.sample_mask] = -1

        result = view_as_windows(self.sample_area, [self.searchKernelSize, self.searchKernelSize, 3], self.windowStep)
        result = result.squeeze()

        axis = (np.zeros((result.shape[0], result.shape[1])) + 1).astype('bool')
        axis *= np.min(result, (2, 3, 4)) >= 0
        index = np.array(range(len(np.where(axis == True)[0])))
        if len(np.where(axis == True)[0]) >= 5000:
            index = np.random.choice(index, 5000, replace=False)

        select_index = np.array([np.where(axis == True)[0][index], np.where(axis == True)[1][index]])
        axis = (np.zeros((result.shape[0], result.shape[1]))).astype('bool')
        axis[select_index[0], select_index[1]] = True

        result = result[axis]

        if self.mirror_hor:
            hor_result = result[:, :, ::-1]  # y轴翻转水平镜像
            result = np.concatenate((result, hor_result))

        if self.mirror_vert:
            vert_result = result[:, ::-1]  # x轴翻转垂直镜像
            result = np.concatenate((result, vert_result))
        return result

    def init_canvas(self):
        # check whether the outputSize adheres to patch+overlap size
        self.num_patches_X = ceil((self.w - self.overlapSize) / (self.patchSize + self.overlapSize))
        self.num_patches_Y = ceil((self.h - self.overlapSize) / (self.patchSize + self.overlapSize))

        # calc needed output image size
        self.required_size_X = self.num_patches_X * self.patchSize + (self.num_patches_X + 1) * self.overlapSize
        self.required_size_Y = self.num_patches_Y * self.patchSize + (self.num_patches_Y + 1) * self.overlapSize

        self.filledMap = np.zeros(
            (self.num_patches_Y, self.num_patches_X))  # map showing which patches have been resolved

        self.up_bound = max(self.y - (self.required_size_Y - self.h) // 2, 0)
        self.down_bound = self.up_bound + self.required_size_Y
        self.left_bound = max(self.x - (self.required_size_X - self.w) // 2, 0)
        self.right_bound = self.left_bound + self.required_size_X

        canvas_with_bound = np.copy(self.img[self.up_bound: self.down_bound, self.left_bound:self.right_bound])
        self.x, self.y = self.left_bound + self.overlapSize, self.up_bound + self.overlapSize
        if self.DEBUG:
            print("num_patches_X %s" % self.num_patches_X)
            print("num_patches_Y %s" % self.num_patches_Y)
            self.canvas_b = np.copy(canvas_with_bound)
            self.canvas_c = np.copy(canvas_with_bound)

        return canvas_with_bound

    def initKDtrees(self):

        topOverlap = self.examplePatches[:, 0:self.overlapSize, :, :]
        leftOverlap = self.examplePatches[:, :, 0:self.overlapSize, :].transpose(0, 2, 1, 3)
        downOverlap = self.examplePatches[:, -self.overlapSize:, :, :]
        rightOverlap = self.examplePatches[:, :, -self.overlapSize:, :].transpose(0, 2, 1, 3)

        self.flatten_combined_ld = np.zeros(
            (leftOverlap.shape[0], leftOverlap.shape[1], leftOverlap.shape[2] * 2, leftOverlap.shape[3]))
        self.flatten_combined_ld[:, :, :leftOverlap.shape[2], :] = leftOverlap
        self.flatten_combined_ld[:, :, leftOverlap.shape[2]:, :] = downOverlap

        self.flatten_combined_ldt = np.zeros(
            (leftOverlap.shape[0], leftOverlap.shape[1], leftOverlap.shape[2] * 3, leftOverlap.shape[3]))
        self.flatten_combined_ldt[:, :, :leftOverlap.shape[2], :] = leftOverlap
        self.flatten_combined_ldt[:, :, leftOverlap.shape[2]:leftOverlap.shape[2] * 2, :] = downOverlap
        self.flatten_combined_ldt[:, :, leftOverlap.shape[2] * 2:, :] = topOverlap

        self.flatten_combined_ldr = np.zeros(
            (leftOverlap.shape[0], leftOverlap.shape[1], leftOverlap.shape[2] * 3, leftOverlap.shape[3]))
        self.flatten_combined_ldr[:, :, :leftOverlap.shape[2], :] = leftOverlap
        self.flatten_combined_ldr[:, :, leftOverlap.shape[2]:leftOverlap.shape[2] * 2, :] = downOverlap
        self.flatten_combined_ldr[:, :, leftOverlap.shape[2] * 2:, :] = rightOverlap

        self.flatten_combined_ldrt = np.zeros(
            (leftOverlap.shape[0], leftOverlap.shape[1], leftOverlap.shape[2] * 4, leftOverlap.shape[3]))
        self.flatten_combined_ldrt[:, :, :leftOverlap.shape[2], :] = leftOverlap
        self.flatten_combined_ldrt[:, :, leftOverlap.shape[2]:leftOverlap.shape[2] * 2, :] = downOverlap
        self.flatten_combined_ldrt[:, :, leftOverlap.shape[2] * 2:leftOverlap.shape[2] * 3, :] = rightOverlap
        self.flatten_combined_ldrt[:, :, leftOverlap.shape[2] * 3:, :] = topOverlap

    def getOverlapAreaTop(self, coord):
        # do I have a top neighbour
        x_range = [(self.patchSize + self.overlapSize) * coord[0],
                   (self.patchSize + self.overlapSize) * (coord[0] + 1) + self.overlapSize]
        y_range = [(self.patchSize + self.overlapSize) * coord[1],
                   (self.patchSize + self.overlapSize) * (coord[1] + 1) + self.overlapSize]

        if self.DEBUG:
            self.canvas_b[x_range[0]:x_range[1], y_range[0]:y_range[1]][0:self.overlapSize, :, :] = 0
        return np.copy(self.canvas_with_bound[x_range[0]:x_range[1], y_range[0]:y_range[1]])[0:self.overlapSize, :, :]

    def getOverlapAreaLeft(self, coord):
        # do I have a left neighbour
        x_range = [(self.patchSize + self.overlapSize) * coord[0],
                   (self.patchSize + self.overlapSize) * (coord[0] + 1) + self.overlapSize]
        y_range = [(self.patchSize + self.overlapSize) * coord[1],
                   (self.patchSize + self.overlapSize) * (coord[1] + 1) + self.overlapSize]

        if self.DEBUG:
            self.canvas_b[x_range[0]:x_range[1], y_range[0]:y_range[1]][:, 0:self.overlapSize, :] = 0
        return np.copy(self.canvas_with_bound[x_range[0]:x_range[1], y_range[0]:y_range[1]])[:, 0:self.overlapSize, :]

    def getOverlapAreaDown(self, coord):
        # do I have a down neighbour
        x_range = [(self.patchSize + self.overlapSize) * coord[0],
                   (self.patchSize + self.overlapSize) * (coord[0] + 1) + self.overlapSize]
        y_range = [(self.patchSize + self.overlapSize) * coord[1],
                   (self.patchSize + self.overlapSize) * (coord[1] + 1) + self.overlapSize]

        if self.DEBUG:
            self.canvas_b[x_range[0]:x_range[1], y_range[0]:y_range[1]][-self.overlapSize:, :, :] = 1
        return np.copy(self.canvas_with_bound[x_range[0]:x_range[1], y_range[0]:y_range[1]])[-self.overlapSize:, :, :]

    def getOverlapAreaRight(self, coord):
        # do I have a left neighbour
        x_range = [(self.patchSize + self.overlapSize) * coord[0],
                   (self.patchSize + self.overlapSize) * (coord[0] + 1) + self.overlapSize]
        y_range = [(self.patchSize + self.overlapSize) * coord[1],
                   (self.patchSize + self.overlapSize) * (coord[1] + 1) + self.overlapSize]

        if self.DEBUG:
            self.canvas_b[x_range[0]:x_range[1], y_range[0]:y_range[1]][:, -self.overlapSize:, :] = 0

        return np.copy(self.canvas_with_bound[x_range[0]:x_range[1], y_range[0]:y_range[1]])[:, -self.overlapSize:, :]

    def linearBlendOverlaps(self, canvasOverlap, examplePatchOverlap, mode):
        if mode == 'left':
            mask = np.repeat(np.arange(self.overlapSize)[np.newaxis, :], np.shape(canvasOverlap)[0],
                             axis=0) / self.overlapSize
        elif mode == 'top':
            mask = np.repeat(np.arange(self.overlapSize)[:, np.newaxis], np.shape(canvasOverlap)[1],
                             axis=1) / self.overlapSize
        elif mode == 'right':
            mask = np.repeat((self.overlapSize - 1 - np.arange(self.overlapSize))[np.newaxis, :],
                             np.shape(canvasOverlap)[0], axis=0) / self.overlapSize
        elif mode == 'down':
            mask = np.repeat((self.overlapSize - 1 - np.arange(self.overlapSize))[:, np.newaxis],
                             np.shape(canvasOverlap)[1], axis=1) / self.overlapSize

        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # cast to 3d array
        return canvasOverlap * (1 - mask) + examplePatchOverlap * mask

    def patchCoord2canvasCoord(self, coord):
        return [(self.patchSize + self.overlapSize) * coord,
                (self.patchSize + self.overlapSize) * (coord + 1) + self.overlapSize]

    def updateCanvas(self, inputPatchId, coord_X, coord_Y):
        # translate Patch coordinate into Canvas coordinate
        x_range = self.patchCoord2canvasCoord(coord_X)
        y_range = self.patchCoord2canvasCoord(coord_Y)
        examplePatch = self.examplePatches[inputPatchId]

        if self.DEBUG:
            print("x_range %s" % x_range)
            print("y_range %s" % y_range)
            print(examplePatch.shape)

        if coord_Y == 0 or self.filledMap[coord_X, coord_Y - 1] == 1:  # left
            # print("have left neighbor")

            canvasOverlap = self.canvas_with_bound[x_range[0]:x_range[1], y_range[0]:y_range[0] + self.overlapSize]
            examplePatchOverlap = np.copy(examplePatch[:, 0:self.overlapSize])
            examplePatch[:, 0:self.overlapSize] = np.copy(
                self.linearBlendOverlaps(canvasOverlap.copy(), examplePatchOverlap.copy(), 'left'))

            if self.DEBUG:
                plt.figure()
                plt.subplot(131);
                plt.imshow(canvasOverlap)
                plt.subplot(132);
                plt.imshow(examplePatchOverlap)
                plt.subplot(133);
                plt.imshow(examplePatch[:, 0:self.overlapSize])

        if coord_X == 0 or self.filledMap[coord_X - 1, coord_Y] == 1:  # top
            # print("have top neighbor")

            canvasOverlap = self.canvas_with_bound[x_range[0]:x_range[0] + self.overlapSize, y_range[0]:y_range[1]]
            examplePatchOverlap = np.copy(examplePatch[0:self.overlapSize, :])
            examplePatch[0:self.overlapSize, :] = np.copy(
                self.linearBlendOverlaps(canvasOverlap, examplePatchOverlap, 'top'))

            if self.DEBUG:
                plt.figure()
                plt.subplot(131);
                plt.imshow(canvasOverlap)
                plt.subplot(132);
                plt.imshow(examplePatchOverlap)
                plt.subplot(133);
                plt.imshow(examplePatch[0:self.overlapSize, :])

        if coord_Y == self.num_patches_X - 1 or self.filledMap[coord_X, coord_Y + 1] == 1:  # right
            # print("have right neighbor")

            canvasOverlap = self.canvas_with_bound[x_range[0]:x_range[1], y_range[1] - self.overlapSize:y_range[1]]
            examplePatchOverlap = np.copy(examplePatch[:, -self.overlapSize:])
            examplePatch[:, -self.overlapSize:] = np.copy(
                self.linearBlendOverlaps(canvasOverlap, examplePatchOverlap, 'right'))

            if self.DEBUG:
                plt.figure()
                plt.subplot(131);
                plt.imshow(canvasOverlap)
                plt.subplot(132);
                plt.imshow(examplePatchOverlap)
                plt.subplot(133);
                plt.imshow(examplePatch[:, -self.overlapSize:])

        if coord_X == self.num_patches_Y - 1 or self.filledMap[coord_X + 1, coord_Y] == 1:  # down
            # print("have down neighbor")

            canvasOverlap = self.canvas_with_bound[x_range[1] - self.overlapSize:x_range[1], y_range[0]:y_range[1]]
            examplePatchOverlap = np.copy(examplePatch[-self.overlapSize:, :])
            examplePatch[-self.overlapSize:, :] = np.copy(
                self.linearBlendOverlaps(canvasOverlap, examplePatchOverlap, 'down'))

            if self.DEBUG:
                plt.figure()
                plt.subplot(131);
                plt.imshow(canvasOverlap)
                plt.subplot(132);
                plt.imshow(examplePatchOverlap)
                plt.subplot(133);
                plt.imshow(examplePatch[-self.overlapSize:, :])

        self.canvas_with_bound[x_range[0]:x_range[1], y_range[0]:y_range[1]] = examplePatch
        if self.DEBUG:
            self.canvas_c[x_range[0]:x_range[1], y_range[0]:y_range[1]] = 0

    def solve_all(self):
        if self.pass_ == True:
            return Image.fromarray(self.img.astype('uint8'))
        else:
            shape = self.getOverlapAreaTop([0, 0]).shape
            patch = np.zeros((shape[0], 0, shape[2]))
            for i in range(self.num_patches_Y):
                for j in range(self.num_patches_X):
                    coord = [self.num_patches_Y - i - 1, j]
                    overlapArea_Top = self.getOverlapAreaTop(coord)
                    overlapArea_Left = self.getOverlapAreaLeft(coord).transpose(1, 0, 2)
                    overlapArea_Right = self.getOverlapAreaRight(coord).transpose(1, 0, 2)
                    overlapArea_Down = self.getOverlapAreaDown(coord)
                    coord_X, coord_Y = coord[0], coord[1]
                    mode = [0, 0, 0, 0]
                    patch = np.zeros((overlapArea_Left.shape[0], 0, overlapArea_Left.shape[2]))
                    if coord_Y == 0 or self.filledMap[coord_X, coord_Y - 1] == 1:  # left
                        mode[0] = 1
                        patch = np.concatenate([patch, overlapArea_Left], axis=1)
                    if coord_X == self.num_patches_Y - 1 or self.filledMap[coord_X + 1, coord_Y] == 1:  # down
                        mode[1] = 1
                        patch = np.concatenate([patch, overlapArea_Down], axis=1)
                    if coord_Y == self.num_patches_X - 1 or self.filledMap[coord_X, coord_Y + 1] == 1:  # right
                        mode[2] = 1
                        patch = np.concatenate([patch, overlapArea_Right], axis=1)
                    if coord_X == 0 or self.filledMap[coord_X - 1, coord_Y] == 1:  # top
                        mode[3] = 1
                        patch = np.concatenate([patch, overlapArea_Top], axis=1)
                    patch = patch[None, :]
                    if mode == [1, 1, 0, 0]:
                        # print('ld')
                        dist = (abs(self.flatten_combined_ld - patch)).sum((1, 2, 3))
                        ind = np.where(dist == dist.min())[0]
                    elif mode == [1, 1, 0, 1]:
                        # print('ldt')
                        dist = (abs(self.flatten_combined_ldt - patch)).sum((1, 2, 3))
                        ind = np.where(dist == dist.min())[0]
                    elif mode == [1, 1, 1, 0]:
                        # print('ldr')
                        dist = (abs(self.flatten_combined_ldr - patch)).sum((1, 2, 3))
                        ind = np.where(dist == dist.min())[0]
                    elif mode == [1, 1, 1, 1]:
                        # print('ldrt')
                        dist = (abs(self.flatten_combined_ldrt - patch)).sum((1, 2, 3))
                        ind = np.where(dist == dist.min())[0]
                    chosenPatchId = ind[0]
                    self.updateCanvas(chosenPatchId, coord[0], coord[1])
                    self.filledMap[coord[0], coord[1]] = 1

            if self.DEBUG:
                plt.figure(figsize=(15, 15))
                plt.subplot(131);
                plt.imshow(self.canvas_with_bound)
                plt.subplot(132);
                plt.imshow(self.canvas_b)
                plt.subplot(133);
                plt.imshow(self.canvas_c)

            self.img[self.up_bound:self.down_bound, self.left_bound:self.right_bound] = self.canvas_with_bound
            self.img[~self.cloth_mask] = self.ori_img[~self.cloth_mask] / 255.
            return Image.fromarray((self.img * 255).astype('uint8'))
