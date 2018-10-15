# -*- coding: utf-8 -*-
import cv2
import numpy as np

def main():

    # マスク画像の読み込み
    diff_mask = cv2.imread("./image/mask_range.png")

    # RGB画像をグレースケールへ
    mask_image_gray = cv2.cvtColor(diff_mask, cv2.COLOR_BGR2GRAY)

    # 出力
    cv2.imwrite("results.png", diff_mask)
    cv2.imshow('window', diff_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
