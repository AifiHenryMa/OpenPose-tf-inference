#! /usr/bin/env python
#coding:utf-8

# 1. 导入python库
import cv2
import numpy as np
import tensorflow as tf
import logging
import scipy.stats as st
import pdb
from tf_pose.pafprocess import pafprocess
from enum import Enum 

# 2. 定义常量
## 2.1 需要测试图片
image_dir = "./images/"
image_name = "hand1.jpg"
image = image_dir + image_name

## 2.2 pb模型文件
model_dir = "./models/graph/cmu/"
model_name = "graph_opt.pb"
model = model_dir + model_name

## 2.3 定义关节点枚举类型
class CocoPart(Enum):
	Nose = 0
	Neck = 1
	RShoulder = 2
	RElbow = 3
	RWrist = 4
	LShoulder = 5
	LElbow = 6
	LWrist = 7
	RHip = 8
	RKnee = 9
	RAnkle = 10
	LHip = 11
	LKnee = 12
	LAnkle = 13
	REye = 14
	LEye = 15
	REar = 16
	LEar = 17
	Background = 18

## 2.4 可能的关键点连线	
CocoPairs = [
    		 (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
    		 (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)] 

## 2.5 关键点显示的颜色
CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

# 3. 定义日志规范（打印调试信息）
logger = logging.getLogger('MyInference')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] [%(message)s]')
ch.setFormatter(formatter)
logger.addHandler(ch)

# 4. 定义函数和类
## 4.1 函数
### 4.1.1 layer函数
def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated

### 4.1.2 estimate_paf函数
def estimate_paf(peaks, heat_mat, paf_mat):
	pafprocess.process_paf(peaks, heat_mat, paf_mat)
	
	humans = []
	print('-------++++++++========' + str(pafprocess.get_num_humans()))
	for human_id in range(pafprocess.get_num_humans()):
		human = Human([])
		is_added = False

		for part_idx in range(18):
			c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
			if c_idx < 0:
				continue

			is_added = True
			human.body_parts[part_idx] = BodyPart(
				'%d-%d' % (human_id, part_idx), part_idx,
				float(pafprocess.get_part_x(c_idx)) / heat_mat.shape[1],
				float(pafprocess.get_part_y(c_idx)) / heat_mat.shape[0],
				pafprocess.get_part_score(c_idx)
			)

		if is_added:
			score = pafprocess.get_score(human_id)
			human.score = score
			humans.append(human)

	return humans

### 4.1.3 draw_humans函数
def draw_humans(npimg, humans, imgcopy=False):
	CocoPairsRender = CocoPairs[:-2]	
	if imgcopy:
		npimg = np.copy(npimg)
	image_h, image_w = npimg.shape[:2]
	centers = {}
	for human in humans:
		# draw point
		for i in range(CocoPart.Background.value):
			if i not in human.body_parts.keys():
				continue

			body_part = human.body_parts[i]
			center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
			centers[i] = center
			cv2.circle(npimg, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)
		
		# draw line
		for pair_order, pair in enumerate(CocoPairsRender):
			if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
				continue

			cv2.line(npimg, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)

	return npimg



## 4.2 类
### 4.2.1 Smoother类
class Smoother(object):
	def __init__(self, inputs, filter_size, sigma):
		self.inputs = inputs
		self.terminals = []
		self.layers = dict(inputs)
		self.filter_size = filter_size
		self.sigma = sigma
		self.setup()

	def setup(self):
		self.feed('data').conv(name='smoothing')

	def get_unique_name(self, prefix):
		ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
		return '%s_%d' % (prefix, ident)

	def feed(self, *args):
		assert len(args) != 0
		self.terminals = []
		for fed_layer in args:
			if isinstance(fed_layer, str):
				try:
					fed_layer = self.layers[fed_layer]
				except KeyError:
					raise KeyError('Unknown layer name fed: %s' % fed_layer)
			self.terminals.append(fed_layer)
		return self

	def gauss_kernel(self, kernlen=21, nsig=3, channels=1):
		interval = (2*nsig+1.)/(kernlen)
		x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
		kern1d = np.diff(st.norm.cdf(x))
		kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
		kernel = kernel_raw/kernel_raw.sum()
		out_filter = np.array(kernel, dtype = np.float32)
		out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
		out_filter = np.repeat(out_filter, channels, axis = 2)
		return out_filter

	def make_gauss_var(self, name, size, sigma, c_i):
		# with tf.device("/cpu:0"):
		kernel = self.gauss_kernel(size, sigma, c_i)
		var = tf.Variable(tf.convert_to_tensor(kernel), name=name)
		return var

	def get_output(self):
		'''Returns the smoother output.'''
		return self.terminals[-1]

	@layer
	def conv(self, input, name, padding='SAME'):
		# Get the number of  channels in the input
		c_i = input.get_shape().as_list()[3]
		# Convolution for a given input and kernel
		convolve = lambda i, k: tf.nn.depthwise_conv2d(i, k, [1, 1, 1, 1], padding=padding)
		with tf.variable_scope(name) as scope:
			kernel = self.make_gauss_var('gauss_weight', self.filter_size, self.sigma, c_i)
			output = convolve(input, kernel)
		return output
	
### 4.2.2 Human类
class Human:
	"""
	body_parts: list of BodyPart
	"""
	__slots__ = ('body_parts', 'pairs', 'uidx_list', 'score')
	
	def __init__(self, pairs):
		self.pairs = []
		self.uidx_list = set()
		self.body_parts = {}
		for pair in pairs:
			self.add_pair(pair)
		self.score = 0.0

	@staticmethod
	def _get_uidx(part_idx, idx):
		return '%d-%d' % (part_idx, idx)

	def add_pair(self, pair):
		self.pairs.append(pair)
		self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1), 
												   pair.part_idx1,
												   pair.coord1[0], pair.coord1[1], pair.score)
		self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
												   pair.part_idx2,
												   pair.coord2[0], pair.coord2[1], pair.score)
		self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
		self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))	
	
	def is_connected(self, other):
		return len(self.uidx_list & other.uidx_list) > 0

	def merge(self, other):
		for pair in other.pairs:
			self.add_pair(pair)

	def part_count(self):
		return len(self.body_parts.keys())

	def get_max_score(self):
		return max([x.score for _, x in self.body_parts.items()])

	def __str__(self):
		return ' '.join([str(x) for x in self.body_parts.values()])

	def __repr__(self):
		return self.__str__()

### 4.2.3 BodyPart类
class BodyPart:
	"""..."""
	__slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')
	
	def __init__(self, uidx, part_idx, x, y, score):
		self.uidx = uidx
		self.part_idx = part_idx
		self.x, self.y = x, y
		self.score = score

	def get_part_name(self):
		return CocoPart(self.part_idx)

	def __str__(self):
		return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' %(self.part_idx, self.x, self.y, self.score)

	def __repr__(self):
		return self.__str__()


# 5. 定义MyTest函数
def MyTest(testimage=image, testmodel=model, outputimage='result2.jpg', upsamp_size=[1312, 736]):

	# 读取图片
	val_image = cv2.imread(testimage, cv2.IMREAD_COLOR)

	# 加载pb模型
	with tf.gfile.GFile(testmodel, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	graph_OpenPose = tf.get_default_graph()
	tf.import_graph_def(graph_def, name='TfPoseEstimator')
	with tf.Session(graph=graph_OpenPose) as sess:
		# 从模型中提取参数
		tensor_image = graph_OpenPose.get_tensor_by_name("TfPoseEstimator/image:0")
		tensor_output = graph_OpenPose.get_tensor_by_name("TfPoseEstimator/Openpose/concat_stage7:0")

		# 有用信息解析
		tensor_heatMat = tensor_output[:,:,:,:19]
		tensor_pafMat = tensor_output[:,:,:,19:]
		upsample_size = tf.placeholder(dtype=tf.int32, shape=(2,), name='upsample_size') # 自己定义
		tensor_heatMat_up = tf.image.resize_area(tensor_heatMat, upsample_size, align_corners=False, name='upsample_heatmat')
		tensor_pafMat_up = tf.image.resize_area(tensor_pafMat, upsample_size, align_corners=False, name='upsample_pafmat')
		smoother = Smoother({'data':tensor_heatMat_up}, 25, 3.0)
		gaussian_heatMat = smoother.get_output()
		max_pooled_in_tensor = tf.nn.pool(gaussian_heatMat, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
		tensor_peaks = tf.where(tf.equal(gaussian_heatMat, max_pooled_in_tensor), 
								gaussian_heatMat,
								tf.zeros_like(gaussian_heatMat))

		# 初始化所有变量
		init = tf.global_variables_initializer()
		sess.run(init)

		# 模型推断
		(peaks, heatMat_up, pafMat_up) = sess.run(
												  [tensor_peaks, tensor_heatMat_up, tensor_pafMat_up],
												  feed_dict={tensor_image:[val_image], upsample_size:upsamp_size}
									         	 )

	peaks = peaks[0]
	heatMat = heatMat_up[0]
	pafMat = pafMat_up[0]

	logger.debug('inference- heatMat=%dx%d pafMat=%dx%d' % (heatMat.shape[1], heatMat.shape[0], pafMat.shape[1], pafMat.shape[0]))

	humans = estimate_paf(peaks, heatMat, pafMat)
	image_output = draw_humans(val_image, humans, imgcopy=False)
		
	print (humans)

	cv2.imwrite(outputimage, image_output)

# 主函数
if __name__ == '__main__':
	MyTest()
	print('Convert Successfully!')


