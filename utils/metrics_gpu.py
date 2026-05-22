import numpy as np
import torch


class Evaluator(object):
    def __init__(self, num_class, device):
        self.num_class = num_class
        self.confusion_matrix = torch.zeros((self.num_class,) * 2).to(device)

    def Pixel_Accuracy(self):
        Acc = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        mAcc = torch.nanmean(Acc)
        return mAcc, Acc

    def Pixel_Precision_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        Pre = self.confusion_matrix[1, 1] / (self.confusion_matrix[0, 1] + self.confusion_matrix[1, 1])
        return Pre

    def Pixel_Recall_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 0] + self.confusion_matrix[1, 1])
        return Rec

    def Pixel_F1_score(self):
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.Pixel_Recall_Rate()
        Pre = self.Pixel_Precision_Rate()
        F1 = 2 * Rec * Pre / (Rec + Pre)
        return F1

    def F1_score_list(self):
        Rec = torch.diag(self.confusion_matrix) / (torch.sum(self.confusion_matrix, axis=0) + 1e-7)
        Pre = torch.diag(self.confusion_matrix) / (torch.sum(self.confusion_matrix, axis=1) + 1e-7)
        F1 = 2 * Rec * Pre / (Rec + Pre + 1e-7)
        return F1

    def Damage_F1(self):
        f1_score_list = self.F1_score_list() + 1e-7
        damage_f1 = (self.num_class)/(1./f1_score_list[0]+1./f1_score_list[1]+1./f1_score_list[2]+1./f1_score_list[3]+1./f1_score_list[4])
        return damage_f1

    # def Overall_F1(self):
    #     return 0.3 * self.Localization_F1() + 0.7 * self.Damage_F1()

    def Mean_Intersection_over_Union(self):
        MIoU = torch.diag(self.confusion_matrix) / (
                torch.sum(self.confusion_matrix, axis=1) + torch.sum(self.confusion_matrix, axis=0) -
                torch.diag(self.confusion_matrix))
        MIoU = torch.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = torch.sum(self.confusion_matrix, axis=1) / torch.sum(self.confusion_matrix)
        iu = torch.diag(self.confusion_matrix) / (
                torch.sum(self.confusion_matrix, axis=1) + torch.sum(self.confusion_matrix, axis=0) -
                torch.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * pre_image[mask].type(torch.uint8) + gt_image[mask].type(torch.uint8)
        count = torch.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_class,) * 2)
