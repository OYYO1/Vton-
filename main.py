import argparse
import os.path
import numpy as np
import cv2 as cv
from numpy import linalg
from common import findFile
from human_parsing import parse_human
print(cv.__version__)
#opencv中的dnn模块
backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_HALIDE, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv.dnn.DNN_BACKEND_OPENCV)#计算后端
targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16, cv.dnn.DNN_TARGET_MYRIAD, cv.dnn.DNN_TARGET_HDDL)#目标设备

parser = argparse.ArgumentParser(description='Use this script to run virtial try-on using CP-VTON', formatter_class=argparse.ArgumentDefaultsHelpFormatter)#用于初始化命令行参数解析器，以便用户可以通过命令行传递参数来配置虚拟试穿过程。
parser.add_argument('--input_image',  type=str, default='test_img/000028_0.jpg', help='Path to image with person.')#人
parser.add_argument('--input_cloth', type=str, default='test_color/000082_1.jpg', help='Path to target cloth image')#衣
parser.add_argument('--gmm_model', '-gmm', default='F:/BaiduNetdiskDownload/virtual_try_on_use_deep_learning/cp_vton_gmm.onnx', help='Path to Geometric Matching Module .onnx model.')#将试穿的服装扭曲以与目标人物对齐的几何匹配模块
parser.add_argument('--tom_model', '-tom', default='F:/BaiduNetdiskDownload/virtual_try_on_use_deep_learning/cp_vton_tom.onnx', help='Path to Try-On Module .onnx model.')#将扭曲的服装与目标人物图像混合的Try-On 模块
parser.add_argument('--segmentation_model', default='F:/BaiduNetdiskDownload/virtual_try_on_use_deep_learning/lip_jppnet_384.pb', help='Path to cloth segmentation .pb model.')#图像分割模型
parser.add_argument('--openpose_proto', default='F:/BaiduNetdiskDownload/virtual_try_on_use_deep_learning/openpose_pose_coco.prototxt', help='Path to OpenPose .prototxt model was trained on COCO dataset.')#人体姿态检测，骨架识别
parser.add_argument('--openpose_model', default='F:/BaiduNetdiskDownload/virtual_try_on_use_deep_learning/openpose_pose_coco.caffemodel', help='Path to OpenPose .caffemodel model was trained on COCO dataset.')#人体姿态检测，骨架识别
parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                    help="Choose one of computation backends: "
                            "%d: automatically (by default), "
                            "%d: Halide language (http://halide-lang.org/), "
                            "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                            "%d: OpenCV implementation" % backends)
parser.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                    help='Choose one of target computation devices: '
                            '%d: CPU target (by default), '
                            '%d: OpenCL, '
                            '%d: OpenCL fp16 (half-float precision), '
                            '%d: NCS2 VPU, '
                            '%d: HDDL VPU' % targets)
args, _ = parser.parse_known_args()

#代码片段主要利用了OpenCV的dnn（深度神经网络）模块来加载和运行深度学习模型。这个模块可以读取各种格式的模型，例如Caffe、TensorFlow、ONNX等，然后在不同的后端上运行，包括CPU、GPU等。
def get_pose_map(image, proto_path, model_path, backend, target, height=256, width=192):
    radius = 5
    inp = cv.dnn.blobFromImage(image, 1.0 / 255, (width, height))

    net = cv.dnn.readNet(proto_path, model_path)
    net.setPreferableBackend(backend)
    net.setPreferableTarget(target)
    net.setInput(inp)
    out = net.forward()

    threshold = 0.1
    _, out_c, out_h, out_w = out.shape
    pose_map = np.zeros((height, width, out_c - 1))
    # last label: Background
    for i in range(0, out.shape[1] - 1):
        heatMap = out[0, i, :, :]
        keypoint = np.full((height, width), -1)
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = width * point[0] // out_w
        y = height * point[1] // out_h
        if conf > threshold and x > 0 and y > 0:
            keypoint[y - radius:y + radius, x - radius:x + radius] = 1
        pose_map[:, :, i] = keypoint
    pose_map = pose_map.transpose(2, 0, 1)
    return pose_map
#使用了双线性插值的方法来平滑处理图像。在预处理系数时，根据输入尺寸和输出尺寸的比例计算出滤波器的大小，并生成双线性插值的权重系数。在水平方向和垂直方向上分别使用这些权重系数对图像进行重采样，最终得到平滑处理后的图像
class BilinearFilter(object):
    """
    PIL bilinear resize implementation
    image = image.resize((image_width // 16, image_height // 16), Image.BILINEAR)
    """
    def _precompute_coeffs(self, inSize, outSize):
        filterscale = max(1.0, inSize / outSize)
        ksize = int(np.ceil(filterscale)) * 2 + 1

        kk = np.zeros(shape=(outSize * ksize, ), dtype=np.float32)
        bounds = np.empty(shape=(outSize * 2, ), dtype=np.int32)

        centers = (np.arange(outSize) + 0.5) * filterscale + 0.5
        bounds[::2] = np.where(centers - filterscale < 0, 0, centers - filterscale)
        bounds[1::2] = np.where(centers + filterscale > inSize, inSize, centers + filterscale) - bounds[::2]
        xmins = bounds[::2] - centers + 1

        points = np.array([np.arange(xmin, xmin + bound) / filterscale for xmin, bound in zip(xmins, bounds[1::2])], dtype=object)
        for xx in range(0, outSize):
            point = points[xx]
            bilinear = np.where(point < 1.0, 1.0 - abs(point), 0.0)
            ww = np.sum(bilinear)
            kk[xx * ksize : xx * ksize + bilinear.size] = np.where(ww == 0.0, bilinear, bilinear / ww)
        return bounds, kk, ksize
#水平方向对图像的重采样
    def _resample_horizontal(self, out, img, ksize, bounds, kk):
        for yy in range(0, out.shape[0]):
            for xx in range(0, out.shape[1]):
                xmin = bounds[xx * 2 + 0]
                xmax = bounds[xx * 2 + 1]
                k = kk[xx * ksize : xx * ksize + xmax]
                out[yy, xx] = np.round(np.sum(img[yy, xmin : xmin + xmax] * k))

# 垂直方向对图像的重采样
    def _resample_vertical(self, out, img, ksize, bounds, kk):
        for yy in range(0, out.shape[0]):
            ymin = bounds[yy * 2 + 0]
            ymax = bounds[yy * 2 + 1]
            k = kk[yy * ksize: yy * ksize + ymax]
            out[yy] = np.round(np.sum(img[ymin : ymin + ymax, 0:out.shape[1]] * k[:, np.newaxis], axis=0))

#图像重采样
    def imaging_resample(self, img, xsize, ysize):
        height, width = img.shape[0:2]
        bounds_horiz, kk_horiz, ksize_horiz = self._precompute_coeffs(width, xsize)
        bounds_vert, kk_vert, ksize_vert    = self._precompute_coeffs(height, ysize)

        out_hor = np.empty((img.shape[0], xsize), dtype=np.uint8)
        self._resample_horizontal(out_hor, img, ksize_horiz, bounds_horiz, kk_horiz)
        out = np.empty((ysize, xsize), dtype=np.uint8)
        self._resample_vertical(out, out_hor, ksize_vert, bounds_vert, kk_vert)
        return out

class CpVton(object):
    def __init__(self, gmm_model, tom_model, backend, target):
        super(CpVton, self).__init__()
        self.gmm_net = cv.dnn.readNet(gmm_model)
        self.tom_net = cv.dnn.readNet(tom_model)
        self.gmm_net.setPreferableBackend(backend)
        self.gmm_net.setPreferableTarget(target)
        self.tom_net.setPreferableBackend(backend)
        self.tom_net.setPreferableTarget(target)
        self.downsample = BilinearFilter()
    def prepare_agnostic(self, segm_image, input_image, pose_map, height=256, width=192):
        palette = {
            'Background'   : (0, 0, 0),
            'Hat'          : (128, 0, 0),
            'Hair'         : (255, 0, 0),
            'Glove'        : (0, 85, 0),
            'Sunglasses'   : (170, 0, 51),
            'UpperClothes' : (255, 85, 0),
            'Dress'        : (0, 0, 85),
            'Coat'         : (0, 119, 221),
            'Socks'        : (85, 85, 0),
            'Pants'        : (0, 85, 85),
            'Jumpsuits'    : (85, 51, 0),
            'Scarf'        : (52, 86, 128),
            'Skirt'        : (0, 128, 0),
            'Face'         : (0, 0, 255),
            'Left-arm'     : (51, 170, 221),
            'Right-arm'    : (0, 255, 255),
            'Left-leg'     : (85, 255, 170),
            'Right-leg'    : (170, 255, 85),
            'Left-shoe'    : (255, 255, 0),
            'Right-shoe'   : (255, 170, 0)
        }
        color2label = {val: key for key, val in palette.items()}#颜色值作为键，语义标签作为值，最终生成一个新的字典
        head_labels = ['Hat', 'Hair', 'Sunglasses', 'Face', 'Pants', 'Skirt']

        segm_image = cv.cvtColor(segm_image, cv.COLOR_BGR2RGB)
        #phead 和 pose_shape 的掩码用于标记头部部位和姿态形状在图像中的位置
        phead = np.zeros((1, height, width), dtype=np.float32)
        pose_shape = np.zeros((height, width), dtype=np.uint8)
        for r in range(height):
            for c in range(width):
                pixel = tuple(segm_image[r, c])
                if tuple(pixel) in color2label:
                    if color2label[pixel] in head_labels:
                        phead[0, r, c] = 1
                    if color2label[pixel] != 'Background':
                        pose_shape[r, c] = 255

        input_image = cv.dnn.blobFromImage(input_image, 1.0 / 127.5, (width, height), mean=(127.5, 127.5, 127.5), swapRB=True)
        input_image = input_image.squeeze(0)
        img_head = input_image * phead - (1 - phead)
        down = self.downsample.imaging_resample(pose_shape, width // 16, height // 16)
        res_shape = cv.resize(down, (width, height), cv.INTER_LINEAR)

        res_shape = cv.dnn.blobFromImage(res_shape, 1.0 / 127.5, mean=(127.5, 127.5, 127.5), swapRB=True)
        res_shape = res_shape.squeeze(0)

        agnostic = np.concatenate((res_shape, img_head, pose_map), axis=0)
        agnostic = np.expand_dims(agnostic, axis=0)
        return agnostic.astype(np.float32)

    def get_warped_cloth(self, cloth_img, agnostic, height=256, width=192):
        cloth = cv.dnn.blobFromImage(cloth_img, 1.0 / 127.5, (width, height), mean=(127.5, 127.5, 127.5), swapRB=True)

        self.gmm_net.setInput(agnostic, "input.1")
        self.gmm_net.setInput(cloth, "input.18")
        theta = self.gmm_net.forward()

        grid = self._generate_grid(theta)#生成一个经过变换后的网格 grid
        warped_cloth = self._bilinear_sampler(cloth, grid).astype(np.float32)
        return warped_cloth

    def get_tryon(self, agnostic, warp_cloth):
        inp = np.concatenate([agnostic, warp_cloth], axis=1)
        self.tom_net.setInput(inp)
        out = self.tom_net.forward()

        p_rendered, m_composite = np.split(out, [3], axis=1)
        p_rendered = np.tanh(p_rendered)
        m_composite = np.exp(-np.logaddexp(0, -m_composite))

        p_tryon = warp_cloth * m_composite + p_rendered * (1 - m_composite)
        rgb_p_tryon = cv.cvtColor(p_tryon.squeeze(0).transpose(1, 2, 0), cv.COLOR_BGR2RGB)
        rgb_p_tryon = (rgb_p_tryon + 1) / 2
        return rgb_p_tryon

    def _compute_L_inverse(self, X, Y):
        N = X.shape[0]

        Xmat = np.tile(X, (1, N))
        Ymat = np.tile(Y, (1, N))
        P_dist_squared = np.power(Xmat - Xmat.transpose(1, 0), 2) + np.power(Ymat - Ymat.transpose(1, 0), 2)

        P_dist_squared[P_dist_squared == 0] = 1
        K = np.multiply(P_dist_squared, np.log(P_dist_squared))

        O = np.ones([N, 1], dtype=np.float32)
        Z = np.zeros([3, 3], dtype=np.float32)
        P = np.concatenate([O, X, Y], axis=1)
        first = np.concatenate((K, P), axis=1)
        second = np.concatenate((P.transpose(1, 0), Z), axis=1)
        L = np.concatenate((first, second), axis=0)
        Li = linalg.inv(L)
        return Li

    def _prepare_to_transform(self, out_h=256, out_w=192, grid_size=5):#为形变过程准备好所需的网格和坐标信息
        grid_X, grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        grid_X = np.expand_dims(np.expand_dims(grid_X, axis=0), axis=3)
        grid_Y = np.expand_dims(np.expand_dims(grid_Y, axis=0), axis=3)

        axis_coords = np.linspace(-1, 1, grid_size)
        N = grid_size ** 2
        P_Y, P_X = np.meshgrid(axis_coords, axis_coords)

        P_X = np.reshape(P_X,(-1, 1))
        P_Y = np.reshape(P_Y,(-1, 1))

        P_X = np.expand_dims(np.expand_dims(np.expand_dims(P_X, axis=2), axis=3), axis=4).transpose(4, 1, 2, 3, 0)
        P_Y = np.expand_dims(np.expand_dims(np.expand_dims(P_Y, axis=2), axis=3), axis=4).transpose(4, 1, 2, 3, 0)
        return grid_X, grid_Y, N, P_X, P_Y
        # grid_X 和 grid_Y 是原始图像的网格坐标。
        # P_X 和 P_Y 是形变后图像的目标网格坐标。
        # N 是网格中点的数量。

    def _expand_torch(self, X, shape):#它的主要作用是确保输入数组的形状与目标形状 shape 兼容或进行扩展。
        if len(X.shape) != len(shape):
            return X.flatten().reshape(shape)
        else:
            axis = [1 if src == dst else dst for src, dst in zip(X.shape, shape)]#这是条件表达式，它的作用是根据 src 是否等于 dst 来确定最终的值。如果 src 等于 dst，则返回 1，否则返回 dst。
            return np.tile(X, axis)

    def  _apply_transformation(self, theta, points, N, P_X, P_Y):
        if len(theta.shape) == 2:
            theta = np.expand_dims(np.expand_dims(theta, axis=2), axis=3)

        batch_size = theta.shape[0]

        P_X_base = np.copy(P_X)
        P_Y_base = np.copy(P_Y)

        Li = self._compute_L_inverse(np.reshape(P_X, (N, -1)), np.reshape(P_Y, (N, -1)))
        Li = np.expand_dims(Li, axis=0)

        # 把theta分解成点坐标
        Q_X = np.squeeze(theta[:, :N, :, :], axis=3)
        Q_Y = np.squeeze(theta[:, N:, :, :], axis=3)

        Q_X += self._expand_torch(P_X_base, Q_X.shape)
        Q_Y += self._expand_torch(P_Y_base, Q_Y.shape)

        points_b = points.shape[0]
        points_h = points.shape[1]
        points_w = points.shape[2]

        P_X = self._expand_torch(P_X, (1, points_h, points_w, 1, N))
        P_Y = self._expand_torch(P_Y, (1, points_h, points_w, 1, N))

        W_X = self._expand_torch(Li[:,:N,:N], (batch_size, N, N)) @ Q_X
        W_Y = self._expand_torch(Li[:,:N,:N], (batch_size, N, N)) @ Q_Y

        W_X = np.expand_dims(np.expand_dims(W_X, axis=3), axis=4).transpose(0, 4, 2, 3, 1)
        W_X = np.repeat(W_X, points_h, axis=1)
        W_X = np.repeat(W_X, points_w, axis=2)

        W_Y = np.expand_dims(np.expand_dims(W_Y, axis=3), axis=4).transpose(0, 4, 2, 3, 1)
        W_Y = np.repeat(W_Y, points_h, axis=1)
        W_Y = np.repeat(W_Y, points_w, axis=2)

        A_X = self._expand_torch(Li[:, N:, :N], (batch_size, 3, N)) @ Q_X
        A_Y = self._expand_torch(Li[:, N:, :N], (batch_size, 3, N)) @ Q_Y

        A_X = np.expand_dims(np.expand_dims(A_X, axis=3), axis=4).transpose(0, 4, 2, 3, 1)
        A_X = np.repeat(A_X, points_h, axis=1)
        A_X = np.repeat(A_X, points_w, axis=2)

        A_Y = np.expand_dims(np.expand_dims(A_Y, axis=3), axis=4).transpose(0, 4, 2, 3, 1)
        A_Y = np.repeat(A_Y, points_h, axis=1)
        A_Y = np.repeat(A_Y, points_w, axis=2)
        #为后续的计算准备了处理好的点集，并确保了它们的形状与其他数组相匹配
        points_X_for_summation = np.expand_dims(np.expand_dims(points[:, :, :, 0], axis=3), axis=4)
        points_X_for_summation = self._expand_torch(points_X_for_summation, points[:, :, :, 0].shape + (1, N))

        points_Y_for_summation = np.expand_dims(np.expand_dims(points[:, :, :, 1], axis=3), axis=4)
        points_Y_for_summation = self._expand_torch(points_Y_for_summation, points[:, :, :, 0].shape + (1, N))
        #对一组点进行变换，根据给定的变换矩阵 A_X, A_Y 和权重矩阵 W_X, W_Y，以及一个参考点 P_X, P_Y，并返回变换后的点的新坐标
        if points_b == 1:
            delta_X = points_X_for_summation - P_X
            delta_Y = points_Y_for_summation - P_Y
        else:
            delta_X = points_X_for_summation - self._expand_torch(P_X, points_X_for_summation.shape)
            delta_Y = points_Y_for_summation - self._expand_torch(P_Y, points_Y_for_summation.shape)

        dist_squared = np.power(delta_X, 2) + np.power(delta_Y, 2)
        dist_squared[dist_squared == 0] = 1
        U = np.multiply(dist_squared, np.log(dist_squared))

        points_X_batch = np.expand_dims(points[:,:,:,0], axis=3)
        points_Y_batch = np.expand_dims(points[:,:,:,1], axis=3)

        if points_b == 1:
            points_X_batch = self._expand_torch(points_X_batch, (batch_size, ) + points_X_batch.shape[1:])
            points_Y_batch = self._expand_torch(points_Y_batch, (batch_size, ) + points_Y_batch.shape[1:])

        points_X_prime = A_X[:,:,:,:,0]+ \
                        np.multiply(A_X[:,:,:,:,1], points_X_batch) + \
                        np.multiply(A_X[:,:,:,:,2], points_Y_batch) + \
                        np.sum(np.multiply(W_X, self._expand_torch(U, W_X.shape)), 4)

        points_Y_prime = A_Y[:,:,:,:,0]+ \
                        np.multiply(A_Y[:,:,:,:,1], points_X_batch) + \
                        np.multiply(A_Y[:,:,:,:,2], points_Y_batch) + \
                        np.sum(np.multiply(W_Y, self._expand_torch(U, W_Y.shape)), 4)

        return np.concatenate((points_X_prime, points_Y_prime), 3)#它将两个数组 points_X_prime 和 points_Y_prime 按照第四个维度（通常是深度或者通道维度）进行拼接，生成一个新的数组

    def _generate_grid(self, theta):
        grid_X, grid_Y, N, P_X, P_Y = self._prepare_to_transform()
        warped_grid = self._apply_transformation(theta, np.concatenate((grid_X, grid_Y), axis=3), N, P_X, P_Y)
        return warped_grid#返回变换后的网格数据

    def _bilinear_sampler(self, img, grid):#双线性插值，每个像素点的插值结果是其周围四个像素点的像素值加权求和，权重由插值时计算得到
        x, y = grid[:,:,:,0], grid[:,:,:,1]

        H = img.shape[2]
        W = img.shape[3]
        max_y = H - 1
        max_x = W - 1

        # 将x和y缩放为[0,W-1/H-1]
        x = 0.5 * (x + 1.0) * (max_x - 1)
        y = 0.5 * (y + 1.0) * (max_y - 1)

        # 为每个(x_i, y_i)取4个最近的角点
        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        # 计算增量，也就是权重
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y  - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y  - y0)

        # 剪辑到范围[0,H-1/W-1]，以免违反图像边界
        x0 = np.clip(x0, 0, max_x)
        x1 = np.clip(x1, 0, max_x)
        y0 = np.clip(y0, 0, max_y)
        y1 = np.clip(y1, 0, max_y)

        #获取角坐标的像素值
        img = img.reshape(-1, H, W)
        Ia = img[:, y0, x0].swapaxes(0, 1)
        Ib = img[:, y1, x0].swapaxes(0, 1)
        Ic = img[:, y0, x1].swapaxes(0, 1)
        Id = img[:, y1, x1].swapaxes(0, 1)

        wa = np.expand_dims(wa, axis=0)
        wb = np.expand_dims(wb, axis=0)
        wc = np.expand_dims(wc, axis=0)
        wd = np.expand_dims(wd, axis=0)

        # compute output
        out = wa*Ia + wb*Ib + wc*Ic + wd*Id#得到插值后的像素值
        return out

class CorrelationLayer(object):
    def __init__(self, params, blobs):
        super(CorrelationLayer, self).__init__()

    def getMemoryShapes(self, inputs):
        fetureAShape = inputs[0]
        b, _, h, w = fetureAShape
        return [[b, h * w, h, w]]

    def forward(self, inputs):
        b, c, h, w = inputs[0].shape
        feature_A = inputs[0].transpose(0, 3, 2, 1)
        feature_B = inputs[1].transpose(0, 2, 3, 1)
        feature_A = feature_A.reshape(1, -1, 512)
        feature_B = feature_B.reshape(1, -1, 512)

        feature_A = feature_A.transpose(0, 2, 1)
        feature_mul = feature_B @ feature_A

        feature_mul = feature_mul.reshape((b, h, w, h * w))
        correlation_tensor = feature_mul.transpose((0, 3, 1, 2))
        correlation_tensor = np.ascontiguousarray(correlation_tensor)
        return [correlation_tensor]

if __name__ == "__main__":
    if not os.path.isfile(args.gmm_model):
        raise OSError("GMM model not exist")
    if not os.path.isfile(args.tom_model):
        raise OSError("TOM model not exist")
    if not os.path.isfile(args.segmentation_model):
        raise OSError("Segmentation model not exist")
    if not os.path.isfile(findFile(args.openpose_proto)):
        raise OSError("OpenPose proto not exist")
    if not os.path.isfile(findFile(args.openpose_model)):
        raise OSError("OpenPose model not exist")

    person_img = cv.imread(args.input_image)

    ratio = 256 / 192
    inp_h, inp_w, _ = person_img.shape
    current_ratio = inp_h / inp_w
    if current_ratio > ratio:
        center_h = inp_h // 2
        out_h = inp_w * ratio
        start = int(center_h - out_h // 2)
        end = int(center_h + out_h // 2)
        person_img = person_img[start:end, ...]
    else:
        center_w = inp_w // 2
        out_w = inp_h / ratio
        start = int(center_w - out_w // 2)
        end = int(center_w + out_w // 2)
        person_img = person_img[:, start:end, :]


    cloth_img = cv.imread(args.input_cloth)

    pose = get_pose_map(person_img, findFile(args.openpose_proto),
                        findFile(args.openpose_model), args.backend, args.target)
    segm_image = parse_human(person_img, args.segmentation_model)
    segm_image = cv.resize(segm_image, (192, 256), cv.INTER_LINEAR)#参数 cv.INTER_LINEAR 表示使用线性插值方法进行图像的缩放，以保持图像质量

    cv.dnn_registerLayer('Correlation', CorrelationLayer)

    model = CpVton(args.gmm_model, args.tom_model, args.backend, args.target)
    agnostic = model.prepare_agnostic(segm_image, person_img, pose)
    warped_cloth = model.get_warped_cloth(cloth_img, agnostic)
    output = model.get_tryon(agnostic, warped_cloth)

    cv.dnn_unregisterLayer('Correlation')
    person_img = cv.imread(args.input_image)
    cloth_img = cv.imread(args.input_cloth)
    cv.imshow('person_img', person_img)
    cv.imshow('cloth_img', cloth_img)

    winName = 'Virtual Try-On'
    cv.namedWindow(winName, cv.WINDOW_NORMAL)
    cv.imshow(winName, output)
    cv.waitKey(0)
    cv.destroyAllWindows()